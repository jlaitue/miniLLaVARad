# Do Clinical VLMs Need Dense Visual Tokens?

Code for measuring how much a LLaVA-style radiology report generator depends on its spatial vision tokens. It trains the same pipeline (BiomedCLIP-CXR encoder, MLP projector, LoRA-tuned Vicuna-7B) with one mean-pooled token and with all 1369 patch tokens, then compares them on MIMIC-CXR, IU X-ray and CheXpert Plus.

It also covers:
- random token sampling before pooling
- image controls: black, another patient's image, a fixed healthy image, COCO and CT
- paired, patient-clustered bootstrap statistics
- generation latency by stage

## Setup

```bash
conda create -n cxr-vlm python=3.11 && conda activate cxr-vlm
pip install -r requirements.txt
cp configs/paths.example.yaml configs/paths.yaml   # then edit
python -m pytest tests
```

The encoder checkpoint (`biomedclipcxr_518_checkpoint.pt`) comes from [microsoft/llava-rad](https://huggingface.co/microsoft/llava-rad). Check its license before use. The optional reference run of the published LLaVA-Rad model also needs `open_clip_torch` and a separate clone, because its license does not allow redistribution:

```bash
git clone https://github.com/microsoft/LLaVA-Rad third_party/LLaVA-Rad
```

## Data

No patient data is included, and you need your own access to each dataset. The reports are under data use agreements: never share manifests, prediction files or score caches derived from them.

| Dataset | Source | Used for |
|---|---|---|
| MIMIC-CXR-JPG and [LLaVA-Rad annotations](https://physionet.org/) | PhysioNet (credentialed) | Training and test |
| IU X-ray | OpenI | Test |
| CheXpert Plus | Stanford AIMI | Test, on a seeded 7.5% sample of patients |

Images are read from flat folders with the shorter side resized to 512 pixels.

```bash
python scripts/prepare_data.py \
    --mimic_annotations /path/to/llava-rad-mimic-cxr-annotations-1.0.0 \
    --iu_reports /path/to/ecgen-radiology --iu_projections /path/to/projections.csv \
    --chexpert_reports /path/to/df_chexpert_plus_240401.csv --chexpert_images /path/to/chexpert/resized/train
```

## Running

```bash
# Train. --tokens is 1, 1369, or 0 for the no-image model (which skips stage 1).
accelerate launch --num_processes 2 --mixed_precision bf16 train.py --stage alignment --tokens 1 --seed 0 --per_gpu_batch 32
accelerate launch --num_processes 2 --mixed_precision bf16 train.py --stage finetune --tokens 1 --seed 0 --per_gpu_batch 64

# Generate. The same manifest order is used for every run.
python predict.py --run_dir runs/T1_seed0/finetune --dataset IUXRAY --out results/predictions/IUXRAY/T1_seed0/original.csv
python predict.py --run_dir runs/T1_seed0/finetune --dataset IUXRAY --sample_tokens 13 --seed 0 --out results/predictions/IUXRAY/T1_seed0/sample_T13_seed0.csv

# Compare. Groups are averaged over their runs; deltas are paired by report and resampled by patient.
python evaluate_runs.py --dataset IUXRAY --baseline T1369 \
    --group T1369=a.csv,b.csv,c.csv --group T1=d.csv,e.csv,f.csv --out results/IUXRAY/t1_vs_t1369.csv

python benchmark_latency.py --run_dir runs/T1_seed0/finetune --out results/latency/T1.json
python scripts/token_redundancy.py --dataset IUXRAY --out results/redundancy/IUXRAY.csv
python scripts/attention_share.py --run_dir runs/T1_seed0/finetune --dataset IUXRAY --predictions <predict output> --out results/attention/T1.csv
```

Training restarts from its latest checkpoint if rerun with the same arguments, and `jobs/make_jobs.py` writes SLURM scripts for the full benchmark without submitting them. Set `XDG_CACHE_HOME` if the CheXbert and RadGraph weights live outside `~/.cache`.

## Evaluation choices

- CheXbert F1 is the primary metric family. The image controls test how much each metric responds to the image.
- "Non-inferior" means the lower bound of the paired 95% CI is above -2 points.
- Every model and dataset uses the encoder's own preprocessing: bicubic resize, center crop and CLIP normalization.

## License

Code license to be added before release. It covers this repository only. LLaVA-Rad, the encoder weights, Vicuna and the datasets keep their own terms.
