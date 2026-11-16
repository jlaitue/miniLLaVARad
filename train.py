"""Train a model: stage 1 aligns the projector, stage 2 fine-tunes the projector and LoRA.

accelerate launch --num_processes 2 train.py --stage alignment --tokens 1 --seed 0 --per_gpu_batch 32
accelerate launch --num_processes 2 train.py --stage finetune --tokens 1 --seed 0 --per_gpu_batch 64
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from accelerate import Accelerator, PartialState
from accelerate.utils import set_seed
from omegaconf import OmegaConf
from transformers import get_cosine_schedule_with_warmup

from cxr_vlm.data import ReportDataset, ResumableSampler, condition_image_paths, load_manifest, make_loader
from cxr_vlm.model import ReportGenerator
from cxr_vlm.paths import REPO_ROOT, load_paths


def trainable_state(model):
    return {name: param.detach().cpu() for name, param in model.named_parameters() if param.requires_grad}


def save_resume_state(path, model, optimizer, scheduler, epoch, batch, update):
    temporary = path.with_suffix(".tmp")
    torch.save({
        "trainable": trainable_state(model), "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
        "epoch": epoch, "batch": batch, "update": update, "rng": torch.get_rng_state(),
    }, temporary)
    temporary.replace(path)


def start_wandb(args, run_dir, config):
    """One W&B run per training stage; a restart from resume.pt continues the same run."""
    import wandb

    id_path = run_dir / "wandb_id.txt"
    if not id_path.exists():
        id_path.write_text(wandb.util.generate_id())
    return wandb.init(
        project=args.wandb_project, entity=args.wandb_entity, id=id_path.read_text().strip(), resume="allow",
        name=f"T{args.tokens}_seed{args.seed}_{args.stage}", group=f"T{args.tokens}_seed{args.seed}",
        job_type=args.stage, dir=str(run_dir), config=config,
    )


@torch.no_grad()
def validation_loss(model, loader, stage, accelerator):
    model.eval()
    total, count = 0.0, 0
    for batch in loader:
        with accelerator.autocast():
            total += model(batch, stage).item() * len(batch["findings"])
        count += len(batch["findings"])
    model.train()
    return total / count


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("alignment", "finetune"), required=True)
    parser.add_argument("--tokens", type=int, choices=(0, 1, 1369), required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--per_gpu_batch", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--max_batches", type=int, help="Stop early; for smoke tests and throughput checks.")
    parser.add_argument("--save_every", type=int, default=1000, help="Updates between resume checkpoints.")
    parser.add_argument("--paths", default=None)
    parser.add_argument("--wandb_project", help="Log every update to this W&B project; off when omitted.")
    parser.add_argument("--wandb_entity")
    args = parser.parse_args()

    if args.stage == "alignment" and args.tokens == 0:
        raise ValueError("The no-image model has no projector to align.")

    paths = load_paths(args.paths)
    config = OmegaConf.load(REPO_ROOT / "configs" / "train.yaml")
    stage = config[args.stage]

    world = PartialState().num_processes
    grad_accum = stage.effective_batch_size // (args.per_gpu_batch * world)
    if grad_accum * args.per_gpu_batch * world != stage.effective_batch_size:
        raise ValueError(f"per_gpu_batch x {world} GPUs must divide {stage.effective_batch_size}.")
    accelerator = Accelerator(gradient_accumulation_steps=grad_accum, mixed_precision="bf16")
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(args.seed)  # Before the model is built, so projector and LoRA init follow the seed.

    run_dir = Path(paths.runs_dir) / f"T{args.tokens}_seed{args.seed}" / args.stage
    resume_path = run_dir / "resume.pt"
    if (run_dir / "run.json").exists():
        accelerator.print(f"{run_dir} is already complete.")
        return
    run_dir.mkdir(parents=True, exist_ok=True)

    projector_path = None
    if args.stage == "finetune" and args.tokens > 0:
        projector_path = Path(paths.runs_dir) / f"T{args.tokens}_seed{args.seed}" / "alignment" / "projector.pt"
        if not projector_path.exists():
            raise FileNotFoundError(f"Run stage 1 first: {projector_path}")
    lora = None
    if args.stage == "finetune":
        lora = {"r": stage.lora_r, "alpha": stage.lora_alpha, "dropout": stage.lora_dropout}
    model = ReportGenerator(args.tokens, paths.encoder_checkpoint, config.llm, config.max_text_length, lora, projector_path)

    model.requires_grad_(False)
    if args.tokens > 0:
        model.projector.requires_grad_(True)
    for name, param in model.llm.named_parameters():
        if "lora_" in name:
            param.requires_grad = True
    parameters = [param for param in model.parameters() if param.requires_grad]

    use_image = args.tokens > 0
    def build_split(split, include_response):
        frame = load_manifest(paths.manifest_dir, "MIMIC_CXR", split)
        image_paths = condition_image_paths(frame, paths.image_dirs.MIMIC_CXR, "original", paths)
        return ReportDataset(frame, image_paths, include_response=include_response, use_image=use_image)

    train_set = build_split("train", include_response=True)
    sampler = ResumableSampler(len(train_set), args.per_gpu_batch, args.seed, accelerator.process_index, world)
    train_loader = make_loader(train_set, args.per_gpu_batch, args.num_workers, sampler=sampler)
    val_loader = make_loader(build_split("val", include_response=True), args.per_gpu_batch, args.num_workers)

    updates_per_epoch = math.ceil(sampler.batches_per_epoch / grad_accum)
    total_updates = updates_per_epoch * stage.epochs
    optimizer = torch.optim.AdamW(parameters, lr=stage.lr, weight_decay=stage.weight_decay)
    scheduler = get_cosine_schedule_with_warmup(optimizer, max(1, int(stage.warmup_ratio * total_updates)), total_updates)
    model, optimizer = accelerator.prepare(model, optimizer)
    unwrapped = accelerator.unwrap_model(model)
    set_seed(args.seed, device_specific=True)  # Distinct LoRA dropout on each GPU.

    start_epoch, start_batch, update = 0, 0, 0
    if resume_path.exists():
        state = torch.load(resume_path, map_location="cpu")
        unwrapped.load_state_dict(state["trainable"], strict=False)
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        start_epoch, start_batch, update = state["epoch"], state["batch"], state["update"]
        accelerator.print(f"Resuming from epoch {start_epoch}, batch {start_batch}, update {update}.")

    accelerator.print(f"{run_dir}: {len(train_set)} reports, {total_updates} updates, accumulation {grad_accum}.")
    wandb_run = None
    if args.wandb_project and accelerator.is_main_process:
        wandb_run = start_wandb(args, run_dir, {
            **OmegaConf.to_container(stage), "llm": config.llm, "max_grad_norm": config.max_grad_norm,
            "tokens": args.tokens, "seed": args.seed, "num_gpus": world, "per_gpu_batch": args.per_gpu_batch,
            "grad_accum": grad_accum, "train_reports": len(train_set), "total_updates": total_updates,
        })
    model.train()
    log_path, started, batches_done = run_dir / "log.jsonl", time.time(), 0
    micro_losses = []
    stop = False
    for epoch in range(start_epoch, stage.epochs):
        sampler.set_state(epoch, start_batch if epoch == start_epoch else 0)
        first_batch = sampler.start_batch
        for index, batch in enumerate(train_loader, start=first_batch):
            with accelerator.accumulate(model):
                loss = model(batch, args.stage)
                accelerator.backward(loss)
                micro_losses.append(loss.detach())
                if accelerator.sync_gradients:
                    grad_norm = accelerator.clip_grad_norm_(parameters, config.max_grad_norm)
                optimizer.step()
                if accelerator.sync_gradients:
                    scheduler.step()
                    update += 1
                optimizer.zero_grad(set_to_none=True)
            batches_done += 1

            if accelerator.sync_gradients:
                # Mean over accumulation steps and GPUs, as the HF Trainer logs it.
                step_loss = accelerator.reduce(torch.stack(micro_losses).mean(), "mean").item()
                micro_losses = []
                record = {
                    "train/loss": step_loss, "train/learning_rate": scheduler.get_last_lr()[0],
                    "train/grad_norm": float(grad_norm), "train/epoch": epoch + (index + 1) / sampler.batches_per_epoch,
                    "perf/seconds_per_batch": (time.time() - started) / batches_done,
                    "perf/peak_gb": torch.cuda.max_memory_allocated() / 2**30,
                }
                if wandb_run:
                    wandb_run.log(record, step=update)
                if accelerator.is_main_process and update % 50 == 0:
                    with log_path.open("a") as log:
                        log.write(json.dumps({"update": update, **record}) + "\n")
                    print({"update": update, **record}, flush=True)
            if accelerator.sync_gradients and update % args.save_every == 0 and accelerator.is_main_process:
                save_resume_state(resume_path, unwrapped, optimizer, scheduler, epoch, index + 1, update)
            if args.max_batches and batches_done >= args.max_batches:
                stop = True
                break
        if stop:
            break
        if accelerator.is_main_process:
            save_resume_state(resume_path, unwrapped, optimizer, scheduler, epoch + 1, 0, update)
            val = validation_loss(unwrapped, val_loader, args.stage, accelerator)
            with log_path.open("a") as log:
                log.write(json.dumps({"epoch": epoch, "update": update, "val_loss": val}) + "\n")
            print({"epoch": epoch, "val_loss": val}, flush=True)
            if wandb_run:
                wandb_run.log({"val/loss": val, "train/epoch": epoch + 1}, step=update)
        accelerator.wait_for_everyone()

    if accelerator.is_main_process:
        if stop:
            print(f"Stopped after {batches_done} batches: "
                  f"{(time.time() - started) / batches_done:.2f} s/batch, "
                  f"{torch.cuda.max_memory_allocated() / 2**30:.1f} GB peak.")
        unwrapped.save(run_dir, {
            "stage": args.stage, "num_vision_tokens": args.tokens, "seed": args.seed, "llm": config.llm,
            "num_gpus": world, "per_gpu_batch": args.per_gpu_batch, "grad_accum": grad_accum,
            "completed_updates": update, "max_batches": args.max_batches,
        })
        resume_path.unlink(missing_ok=True)
        if wandb_run:
            wandb_run.finish()
    accelerator.wait_for_everyone()


if __name__ == "__main__":
    main()
