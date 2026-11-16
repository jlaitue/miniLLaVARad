"""Compare groups of prediction files on one dataset.

python evaluate_runs.py --dataset MIMIC_CXR --baseline T1 \
    --group T1=a/original.csv \
    --group T13=a/sample_T13_seed0.csv,a/sample_T13_seed1.csv --out results/MIMIC_CXR/t13_vs_t1.csv
"""

import argparse
from pathlib import Path

from cxr_vlm.metrics import load_predictions, paired_bootstrap, score_reports


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--baseline", required=True)
    parser.add_argument("--group", action="append", required=True, help="name=file.csv,file.csv")
    parser.add_argument("--out", required=True)
    parser.add_argument("--n_bootstrap", type=int, default=5000)
    parser.add_argument("--margin", type=float, default=2.0, help="Non-inferiority margin in points.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    groups, reference = {}, None
    for item in args.group:
        name, files = item.split("=", 1)
        groups[name] = []
        for csv_path in files.split(","):
            frame = load_predictions(csv_path)
            key = frame[["patient_id", "study_id", "image_id", "target"]]
            if reference is None:
                reference = key
            elif not key.equals(reference):
                raise ValueError(f"{csv_path} is not aligned with the other runs.")
            groups[name].append(score_reports(frame, Path(csv_path).with_suffix(".scores.npz")))
    if args.baseline not in groups:
        raise ValueError(f"--baseline {args.baseline} is not one of {sorted(groups)}")

    summary, n_patients = paired_bootstrap(
        groups, reference["patient_id"].to_numpy(), args.baseline, args.n_bootstrap, seed=args.seed, margin=args.margin
    )
    summary.insert(0, "dataset", args.dataset)
    summary["n_reports"], summary["n_patients"] = len(reference), n_patients
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(f"Saved {out} ({len(reference)} reports, {n_patients} patients)")


if __name__ == "__main__":
    main()
