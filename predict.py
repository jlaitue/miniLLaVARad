"""Generate reports for one dataset and condition from a trained model."""

import argparse
import json
from pathlib import Path

import pandas as pd
import torch
from tqdm import tqdm

from cxr_vlm.data import IMAGE_CONDITIONS, ReportDataset, condition_image_paths, load_manifest, make_loader
from cxr_vlm.model import load_trained
from cxr_vlm.paths import load_paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True, help="Directory of a finished stage 2 run.")
    parser.add_argument("--dataset", required=True, choices=("MIMIC_CXR", "IUXRAY", "CHEXPERT"))
    parser.add_argument("--condition", default="original", choices=IMAGE_CONDITIONS)
    parser.add_argument("--no_indication", action="store_true")
    parser.add_argument("--sample_tokens", type=int, help="Random patches kept per image before pooling (T=1 model).")
    parser.add_argument("--ablate_projector", action="store_true", help="Zero the projected vision tokens.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for token sampling and control image assignment.")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--limit", type=int, help="Use only the first rows; for smoke tests.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--paths", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"{out} already exists.")
        return
    paths = load_paths(args.paths)
    model, metadata = load_trained(args.run_dir, paths.encoder_checkpoint)

    frame = load_manifest(paths.manifest_dir, args.dataset, "test")
    if args.limit:
        frame = frame.head(args.limit)
    image_paths = condition_image_paths(frame, paths.image_dirs[args.dataset], args.condition, paths, args.seed)
    dataset = ReportDataset(
        frame, image_paths, use_image=metadata["num_vision_tokens"] > 0, use_indication=not args.no_indication
    )
    generator = torch.Generator(device="cuda").manual_seed(args.seed)

    rows = []
    for batch in tqdm(make_loader(dataset, args.batch_size, args.num_workers), desc=out.parent.name):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            reports = model.generate_reports(batch, args.max_new_tokens, args.sample_tokens, generator, args.ablate_projector)
        rows.extend(zip(batch["patient_id"], batch["study_id"], batch["image_id"], reports, batch["findings"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["patient_id", "study_id", "image_id", "prediction", "target"]
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps({**vars(args), **metadata}, indent=2))


if __name__ == "__main__":
    main()
