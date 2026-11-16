# Project handoff

Full paper from the ICML 2026 MechInterp workshop paper "Do Clinical VLMs Need Dense Visual Tokens?" (non-archival). Likely venue is CHIL; the 2027 call was not posted as of 2026-11-02. Old code: `../meduq`, kept as is with the user's uncommitted changes. This repo replaces it, and nothing from old runs is reused.

## Decisions by the user
- This is an empirical findings paper about LLaVA-like pipelines, not a claim of a stronger model. Mixed results across datasets are fine and must be reported.
- No coarse-region experiments and no radiologist study.
- Everything is retrained and rerun from scratch.
- One training seed per model (2026-11-02). The reviewers asked for multiple seeds and CIs on the random token sampling, not retraining. Sampling keeps 5 seeds. CIs therefore cover patients and token draws, not training variance, and the paper must say so.
- Timeline: about 2 months of running is acceptable (user, 2026-11-02).
- Release of MIMIC-trained checkpoints is parked until the end. Code license is also still open.
- Autonomy (2026-11-16): the user is testing how far AI can fully automate this project, and Claude drives it end to end, including submitting SLURM jobs without per-job approval. Ask only when the user flags an action or a decision is theirs (framing, venue, release, license).
- SLURM: QoS caps are group-wide GPU counts (checked 2026-11-16): `-q shamout` = 6 GPUs shared with colleagues, often full; `-q cair` = 2 GPUs. They are not tied to particular nodes.

## Bugs in the old pipeline, fixed here
- T=1 used ImageNet normalization and bilinear resize; the encoder expects CLIP normalization and bicubic.
- No same-pipeline T=1369 model existed, and generation always mean-pooled.
- `hypers.seed` was applied after model build, so all "seeds" shared init; every config used seed 0.
- Token sampling floored counts (1% is T=13, not 14), and used one fixed draw per batch from a hardcoded seed.
- Cosine LR schedule ended halfway on multi-GPU runs. Stage 1 weight decay and batch size differed from LLaVA-Rad.
- CheXpert prompts contained "indication: None", and 52 IU prompts contained "indication: nan".
- CheXpert split came from an unseeded row sample. RadGraph failures were silently scored 0.00.
- Bootstrap resampled rows rather than patients, with no paired test. MIMIC test has 285 patients.
- The old `healthy_cxr/predictions.csv` targets were all overwritten with one report; never cite those scores.
- LLaVA-Rad's license forbids redistribution, so it is not vendored. `cxr_vlm/encoder.py` is verified identical on tensors (max diff 0.0).

## Verified on 2026-11-02 (H100, llavarad conda env)
Manifests match the old data (MIMIC test 2461/2461, IU 3199/3199). Batch building matches the old model tensor for tensor at T=1 and T=1369. 14 unit tests pass. Short runs of every script passed on 1 GPU, as did restart from checkpoint. Not yet run: multi-GPU (`00_smoke_2gpu.sbatch`), full-scale jobs, and the benchmark on A100.

## Timing on one H100
| Run | Batch | s/batch | Peak GB | Single-GPU hours |
|---|---|---|---|---|
| T=1 stage 1 | 64 | 2.17 | 26 | 3.8 |
| T=1 stage 2 | 64 | 3.34 | 33 | 17 |
| T=0 stage 2 | 64 | 3.23 | 33 | 17 |
| T=1369 stage 1 | 16 | 5.34 | 43 | 37 |
| T=1369 stage 2 | 16 | 6.56 | 47 | 137 |

T=1369 at batch 32 ran out of memory on the 95 GB H100.

## Run order
User's order (2026-11-16): cheap models first, T=1369 last. Goal is 3 fresh checkpoints (T=1, T=0, T=1369).
1. `jobs/generated/00_smoke_2gpu.sbatch`
2. T=1 seed 0 (stage 1 then stage 2) and T=0 seed 0 (stage 2 only).
3. Main results without T=1369: all T=1 and T=0 predictions on the 3 datasets, including the random token sampling (sampling uses the T=1 checkpoint) and the LLaVA-Rad reference runs: `predict_array_main.sbatch` (188 tasks), then `evaluate_runs.py`.
4. T=1369 seed 0, stage 1 then stage 2. Check it comes close to the published LLaVA-Rad scores, then `predict_array_T1369.sbatch` (23 tasks).
5. `latency.sbatch` (T=1 vs T=1369), `scripts/token_redundancy.py`, `scripts/attention_share.py`.
6. Still to write: figures and tables, aggregate OOD finding counts (Reviewer 1), and the paper. Fix the LoRA citation (Hu et al.).

## Current state (2026-11-16)
Smoke tests passed on 2xA100 80GB (cair) for both stages of T=1 and T=1369; W&B logging verified (entity `foundation_uq`, projects `clinical-vlm-tokens` and `clinical-vlm-tokens-smoke`). A100 speeds: T=1 stage 1 4.3 s/batch (28 GB), T=1 stage 2 8.1 s/batch (34 GB), T=1369 stage 1 9.1 s/batch (45 GB), T=1369 stage 2 11.7 s/batch at per-GPU 16 (51 GB). Submitted on `-q cair` as a chain: T=1 alignment 19395853 -> T=1 finetune 19395854 (afterok) -> T=0 finetune 19395855 (afterany). Data counts were confirmed by the user to match LLaVA-Rad (400,042 train reports). Regenerate scripts with `python jobs/make_jobs.py --hf_home ... --cache_home ...`. Full transcript of the earlier session: `~/.claude/projects/-scratch-ljl5178-project-beta-meduq/66f2ad74-2cbc-442a-99d9-0f9e8cd8cb14.jsonl`.

Reviews are in `../meduq/mech_interp_icml_reviews.md`. Do not print MIMIC or CheXpert report text in a session: those datasets are under data use agreements.
