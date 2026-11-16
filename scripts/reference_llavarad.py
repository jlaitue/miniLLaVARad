"""Run the published LLaVA-Rad checkpoint on our manifests, as an external reference.

LLaVA-Rad's license does not allow redistribution, so clone it separately and put it on the path:
    git clone https://github.com/microsoft/LLaVA-Rad third_party/LLaVA-Rad
    PYTHONPATH=third_party/LLaVA-Rad python scripts/reference_llavarad.py --dataset MIMIC_CXR --out ...
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

# LLaVA-Rad's checkpoints predate PyTorch's weights_only default. They come from Microsoft's own repository.
_torch_load = torch.load
torch.load = lambda *args, **kwargs: _torch_load(*args, **{**kwargs, "weights_only": False})

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from cxr_vlm.data import IMAGE_CONDITIONS, IMAGE_SIZE, build_prompt, condition_image_paths, load_manifest
from cxr_vlm.paths import load_paths
from llava.constants import IMAGE_TOKEN_INDEX
from llava.mm_utils import KeywordsStoppingCriteria, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=("MIMIC_CXR", "IUXRAY", "CHEXPERT"))
    parser.add_argument("--condition", default="original", choices=IMAGE_CONDITIONS)
    parser.add_argument("--no_indication", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--out", required=True)
    parser.add_argument("--paths", default=None)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        print(f"{out} already exists.")
        return
    paths = load_paths(args.paths)
    frame = load_manifest(paths.manifest_dir, args.dataset, "test")
    if args.limit:
        frame = frame.head(args.limit)
    image_paths = condition_image_paths(frame, paths.image_dirs[args.dataset], args.condition, paths, args.seed)

    disable_torch_init()
    tokenizer, model, image_processor, _ = load_pretrained_model("microsoft/llava-rad", "lmsys/vicuna-7b-v1.5", "llavarad")

    rows = []
    for index, row in tqdm(frame.iterrows(), total=len(frame)):
        path = image_paths[index]
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)) if path is None else Image.open(path).convert("RGB")
        pixels = image_processor.preprocess(image, return_tensors="pt")["pixel_values"][0].half().unsqueeze(0).cuda()
        prompt = build_prompt(row["indication"], use_indication=not args.no_indication)
        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()
        stopping = KeywordsStoppingCriteria(["</s>"], tokenizer, input_ids)
        with torch.inference_mode():
            output = model.generate(
                input_ids, images=pixels, do_sample=False, max_new_tokens=args.max_new_tokens,
                use_cache=True, stopping_criteria=[stopping],
            )
        report = tokenizer.batch_decode(output[:, input_ids.shape[1]:], skip_special_tokens=True)[0].strip()
        rows.append((row["patient_id"], row["study_id"], row["image"], report, row["findings"]))

    out.parent.mkdir(parents=True, exist_ok=True)
    columns = ["patient_id", "study_id", "image_id", "prediction", "target"]
    pd.DataFrame(rows, columns=columns).to_csv(out, index=False)
    out.with_suffix(".json").write_text(json.dumps(vars(args), indent=2))


if __name__ == "__main__":
    main()
