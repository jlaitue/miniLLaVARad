"""Write the sbatch scripts and task lists for the whole benchmark. This never submits anything."""

import argparse
from pathlib import Path

GENERATED = Path(__file__).resolve().parent / "generated"
REPO = Path(__file__).resolve().parent.parent
TRAIN_SEEDS = (0,)
SAMPLING_SEEDS = (0, 1, 2, 3, 4)
SAMPLE_BUDGETS = (1, 2, 4, 8, 13, 34, 68, 137, 342, 684)
DATASETS = ("MIMIC_CXR", "IUXRAY", "CHEXPERT")
CONTROLS = ("black", "shuffled", "healthy", "coco", "ct")
PER_GPU_BATCH = {("alignment", 1): 64, ("alignment", 1369): 16, ("finetune", 0): 64, ("finetune", 1): 64, ("finetune", 1369): 16}
PREDICT_BATCH = {0: 32, 1: 32, 1369: 8}
# (GPUs, walltime). T=1369 needs about 2 to 4 times the T=1 time per report.
RESOURCES = {("alignment", 1): (2, "1-00:00:00"), ("alignment", 1369): (4, "2-00:00:00"),
             ("finetune", 0): (2, "2-00:00:00"), ("finetune", 1): (2, "2-00:00:00"), ("finetune", 1369): (4, "4-00:00:00")}


def header(args, name, gpus, time, array=None):
    lines = [
        "#!/bin/bash", f"#SBATCH -J {name}", "#SBATCH -p nvidia", f"#SBATCH -q {args.qos}",
        f"#SBATCH --gres=gpu:{args.gpu}:{gpus}", f"#SBATCH --constraint={args.constraint}" if args.constraint else "",
        f"#SBATCH --time={time}", f"#SBATCH --cpus-per-task={8 * gpus}", f"#SBATCH --mem={80 * gpus}GB",
        f"#SBATCH -o logs/{name}.%A_%a.out" if array else f"#SBATCH -o logs/{name}.%j.out",
        f"#SBATCH --array={array}" if array else "",
        "set -euo pipefail", f"cd {REPO}", "set +u  # conda activate.d hooks read unset variables", 'eval "$(conda shell.bash hook)"', f"conda activate {args.conda_env}", "set -u",
        f"export HF_HOME={args.hf_home} XDG_CACHE_HOME={args.cache_home} TOKENIZERS_PARALLELISM=false", "",
    ]
    return "\n".join(line for line in lines if line != "" or line is lines[-1]) + "\n"


def write(name, text):
    path = GENERATED / name
    path.write_text(text)
    return path


def train_command(stage, tokens, seed, gpus, extra=""):
    batch = PER_GPU_BATCH[(stage, tokens)]
    return (f"accelerate launch --num_processes {gpus} --mixed_precision bf16 train.py --stage {stage} "
            f"--tokens {tokens} --seed {seed} --per_gpu_batch {batch}{extra}\n")


def predict_tasks():
    """One line per generation run. Each model is trained once; variance comes from sampling seeds and patients."""
    tasks = []

    def add(tokens, seed, dataset, name, flags=""):
        tasks.append(
            f"python predict.py --run_dir runs/T{tokens}_seed{seed}/finetune --dataset {dataset} "
            f"--batch_size {PREDICT_BATCH[tokens]} {flags} --out results/predictions/{dataset}/T{tokens}_seed{seed}/{name}.csv"
        )

    for dataset in DATASETS:
        for tokens in (1, 1369):
            for seed in TRAIN_SEEDS:
                add(tokens, seed, dataset, "original")
            for control in CONTROLS:
                add(tokens, 0, dataset, control, f"--condition {control}")
            add(tokens, 0, dataset, "ablated_projector", "--ablate_projector")
            if dataset != "CHEXPERT":  # CheXpert Plus has no indications.
                add(tokens, 0, dataset, "no_indication", "--no_indication")
        add(0, 0, dataset, "original")
        for budget in SAMPLE_BUDGETS:
            for seed in SAMPLING_SEEDS:
                add(1, 0, dataset, f"sample_T{budget}_seed{seed}", f"--sample_tokens {budget} --seed {seed}")
        for condition in ("original", "black", "shuffled", "healthy"):
            tasks.append(
                f"PYTHONPATH=third_party/LLaVA-Rad python scripts/reference_llavarad.py --dataset {dataset} "
                f"--condition {condition} --out results/predictions/{dataset}/llavarad_reference/{condition}.csv"
            )
    return tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--qos", default="shamout")
    parser.add_argument("--gpu", default="a100")
    parser.add_argument("--constraint", default="80g")
    parser.add_argument("--conda_env", default="llavarad")
    parser.add_argument("--hf_home", required=True)
    parser.add_argument("--cache_home", required=True, help="XDG cache that holds the CheXbert and RadGraph weights.")
    args = parser.parse_args()
    GENERATED.mkdir(exist_ok=True)
    (REPO / "logs").mkdir(exist_ok=True)

    write("00_smoke_2gpu.sbatch", header(args, "smoke_2gpu", 2, "02:00:00") + "\n".join([
        "# Distributed checks with a throwaway runs folder: both stages for T=1 and T=1369.",
        "sed 's#^runs_dir:.*#runs_dir: runs_smoke#' configs/paths.yaml > configs/paths.smoke.yaml",
        *[train_command(stage, tokens, 99, 2, " --max_batches 40 --save_every 1 --paths configs/paths.smoke.yaml").strip()
          for tokens in (1, 1369) for stage in ("alignment", "finetune")],
    ]) + "\n")

    for tokens in (1, 1369):
        for seed in TRAIN_SEEDS:
            for stage in ("alignment", "finetune"):
                gpus, time = RESOURCES[(stage, tokens)]
                write(f"train_T{tokens}_seed{seed}_{stage}.sbatch",
                      header(args, f"T{tokens}s{seed}_{stage[:2]}", gpus, time) + train_command(stage, tokens, seed, gpus))
    gpus, time = RESOURCES[("finetune", 0)]
    write("train_T0_seed0_finetune.sbatch", header(args, "T0s0_fi", gpus, time) + train_command("finetune", 0, 0, gpus))

    # T=1369 is trained last, so its generation runs get their own array.
    tasks = predict_tasks()
    groups = {"main": [t for t in tasks if "/T1369_seed" not in t], "T1369": [t for t in tasks if "/T1369_seed" in t]}
    for group, lines in groups.items():
        task_file = f"predict_tasks_{group}.txt"
        (GENERATED / task_file).write_text("\n".join(lines) + "\n")
        runner = f'sed -n "${{SLURM_ARRAY_TASK_ID}}p" jobs/generated/{task_file} | bash -euo pipefail\n'
        write(f"predict_array_{group}.sbatch",
              header(args, f"predict_{group}", 1, "06:00:00", f"1-{len(lines)}%{args_limit}") + runner)

    latency = "".join(
        f"python benchmark_latency.py --run_dir runs/T{tokens}_seed0/finetune --out results/latency/T{tokens}.json\n"
        for tokens in (1, 1369)
    )
    write("latency.sbatch", header(args, "latency", 1, "03:00:00") + latency)
    print(f"Wrote {len(list(GENERATED.iterdir()))} files to {GENERATED}; {len(tasks)} generation runs.")


args_limit = 8  # Concurrent array tasks, to leave room for colleagues on shared nodes.

if __name__ == "__main__":
    main()
