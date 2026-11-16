"""How much does a random subset of patches change the pooled token?

Compares the cosine between a subset's mean and the full mean of the same image against the
cosine between full means of different patients' images, which gives the scale for reading it.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxr_vlm.data import ReportDataset, condition_image_paths, load_manifest, make_loader
from cxr_vlm.encoder import VisionEncoder
from cxr_vlm.model import pool_tokens
from cxr_vlm.paths import load_paths


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--budgets", type=int, nargs="+", default=[1, 2, 4, 8, 13, 34, 68, 137, 342, 684])
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paths", default=None)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    encoder = VisionEncoder(paths.encoder_checkpoint).cuda()
    frame = load_manifest(paths.manifest_dir, args.dataset, "test")
    dataset = ReportDataset(frame, condition_image_paths(frame, paths.image_dirs[args.dataset], "original", paths))

    means, same_image = [], {(seed, budget): [] for seed in args.seeds for budget in args.budgets}
    generators = {seed: torch.Generator(device="cuda").manual_seed(seed) for seed in args.seeds}
    for batch in make_loader(dataset, args.batch_size):
        features = encoder(batch["image"].cuda())
        full = features.mean(dim=1)
        means.append(full.cpu())
        for seed, generator in generators.items():
            for budget in args.budgets:
                sampled = pool_tokens(features, 1, budget, generator).squeeze(1)
                same_image[(seed, budget)].append(F.cosine_similarity(sampled, full, dim=-1).cpu())

    means = F.normalize(torch.cat(means), dim=-1)
    patients = frame["patient_id"].to_numpy()
    rng = np.random.default_rng(0)
    first, second = rng.integers(len(means), size=(2, 200_000))
    keep = patients[first] != patients[second]
    between = (means[first[keep]] * means[second[keep]]).sum(-1).numpy()

    rows = [{"dataset": args.dataset, "budget": budget, "seed": seed, "cosine_to_full": torch.cat(values).mean().item()}
            for (seed, budget), values in same_image.items()]
    summary = pd.DataFrame(rows)
    summary["between_patient_cosine_mean"] = between.mean()
    summary["between_patient_cosine_p95"] = np.quantile(between, 0.95)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.groupby("budget")["cosine_to_full"].agg(["mean", "std"]).to_string())
    print(f"Between patients: mean {between.mean():.4f}, p95 {np.quantile(between, 0.95):.4f}")


if __name__ == "__main__":
    main()
