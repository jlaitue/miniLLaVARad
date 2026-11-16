"""Time each generation stage for a trained model on one GPU, without metrics or data loading.

Every run decodes a fixed number of tokens with no early stop, so models are directly comparable.
Encoder, prefill and decode are timed separately because the encoder cost does not depend on T.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cxr_vlm.data import ReportDataset, condition_image_paths, load_manifest, make_loader
from cxr_vlm.model import load_trained, pool_tokens
from cxr_vlm.paths import load_paths


def timed(function):
    torch.cuda.synchronize()
    start = time.perf_counter()
    output = function()
    torch.cuda.synchronize()
    return output, time.perf_counter() - start


@torch.no_grad()
def run_batch(model, batch, new_tokens):
    llm, images = model.llm, batch["image"].to("cuda")
    features, encoder = timed(lambda: model.encoder(images))

    def assemble():
        tokens = pool_tokens(features, model.num_vision_tokens)
        projected = model.projector(tokens.to(dtype=next(model.projector.parameters()).dtype))
        return model.build_chat_batch(batch["prompt"], projected, inference=True)

    (embeds, mask, _), assembly = timed(assemble)
    output, prefill = timed(lambda: llm(inputs_embeds=embeds, attention_mask=mask, use_cache=True))

    def decode():
        past, current, attention = output.past_key_values, output.logits[:, -1].argmax(-1, keepdim=True), mask
        for _ in range(new_tokens - 1):
            attention = torch.cat([attention, attention.new_ones(attention.shape[0], 1)], dim=1)
            step = llm(input_ids=current, attention_mask=attention, past_key_values=past, use_cache=True)
            past, current = step.past_key_values, step.logits[:, -1].argmax(-1, keepdim=True)

    _, decoding = timed(decode)
    return {"encoder_s": encoder, "assembly_s": assembly, "prefill_s": prefill,
            "per_token_ms": 1000 * decoding / (new_tokens - 1), "input_length": embeds.shape[1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run_dir", required=True)
    parser.add_argument("--batch_sizes", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--batches", type=int, default=30)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--new_tokens", type=int, default=128)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paths", default=None)
    args = parser.parse_args()

    paths = load_paths(args.paths)
    model, metadata = load_trained(args.run_dir, paths.encoder_checkpoint)
    frame = load_manifest(paths.manifest_dir, "MIMIC_CXR", "test")
    dataset = ReportDataset(frame, condition_image_paths(frame, paths.image_dirs.MIMIC_CXR, "original", paths))

    results = {"gpu": torch.cuda.get_device_name(0), "num_vision_tokens": metadata["num_vision_tokens"], "runs": []}
    for batch_size in args.batch_sizes:
        torch.cuda.reset_peak_memory_stats()
        records = []
        for index, batch in enumerate(make_loader(dataset, batch_size, num_workers=4)):
            if index >= args.batches + args.warmup:
                break
            with torch.autocast("cuda", dtype=torch.bfloat16):
                record = run_batch(model, batch, args.new_tokens)
            if index >= args.warmup:
                records.append(record)
        summary = {key: {"mean": float(np.mean([r[key] for r in records])), "sd": float(np.std([r[key] for r in records]))}
                   for key in records[0]}
        report_seconds = (summary["encoder_s"]["mean"] + summary["assembly_s"]["mean"] + summary["prefill_s"]["mean"]
                          + (args.new_tokens - 1) * summary["per_token_ms"]["mean"] / 1000)
        results["runs"].append({
            "batch_size": batch_size, "timing": summary, "seconds_per_batch": report_seconds,
            "reports_per_second": batch_size / report_seconds, "peak_vram_gb": torch.cuda.max_memory_allocated() / 2**30,
        })
        print(json.dumps(results["runs"][-1], indent=2), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
