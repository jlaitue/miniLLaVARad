"""Internal probe: how much attention do report tokens pay to the vision token positions?

Replays each generated report through the LLM and measures, per layer, the attention from report
tokens to vision positions, relative to what a uniform split over the context would give.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxr_vlm.data import IMAGE_TOKEN, ReportDataset, condition_image_paths, load_manifest, make_loader
from cxr_vlm.model import load_trained
from cxr_vlm.paths import load_paths


@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True, help="predict.py output for this run and dataset.")
    parser.add_argument("--reports", type=int, default=200)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paths", default=None)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    model, metadata = load_trained(args.run_dir, paths.encoder_checkpoint)
    num_tokens = metadata["num_vision_tokens"]
    predictions = pd.read_csv(args.predictions).head(args.reports)
    frame = load_manifest(paths.manifest_dir, args.dataset, "test").head(len(predictions))
    dataset = ReportDataset(frame, condition_image_paths(frame, paths.image_dirs[args.dataset], "original", paths))

    rows = []
    for index, batch in enumerate(make_loader(dataset, 1, num_workers=2)):
        prompt = batch["prompt"][0]
        report = str(predictions["prediction"].iloc[index])
        left, _ = prompt.split(IMAGE_TOKEN)
        vision_start = 1 + len(model._tokenize(left))
        full_prompt = f"{prompt} {report}</s>"
        with torch.autocast("cuda", dtype=torch.bfloat16):
            embeds, mask, labels = model.build_chat_batch([full_prompt], model.image_embeds(batch["image"]))
            output = model.llm(inputs_embeds=embeds, attention_mask=mask, output_attentions=True, use_cache=False)
        report_positions = (labels[0] != -100).nonzero().squeeze(-1)
        for layer, attention in enumerate(output.attentions):
            to_vision = attention[0].float().mean(0)[report_positions, vision_start : vision_start + num_tokens].sum(-1)
            uniform = num_tokens / (report_positions.float() + 1)
            rows.append({
                "report": index, "layer": layer, "attention_to_vision": to_vision.mean().item(),
                "relative_to_uniform": (to_vision / uniform).mean().item(),
            })

    summary = pd.DataFrame(rows)
    summary.insert(0, "num_vision_tokens", num_tokens)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    print(summary.groupby("layer")[["attention_to_vision", "relative_to_uniform"]].mean().to_string())


if __name__ == "__main__":
    main()
