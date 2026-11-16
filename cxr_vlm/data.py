"""Manifests, prompts, image conditions and dataloaders.

A manifest is a CSV with columns patient_id, study_id, image, indication, findings, split,
written by scripts/prepare_data.py. MIMIC-CXR's test split only holds rule-based reports.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Sampler
from torchvision import transforms

IMAGE_SIZE = 518
# BiomedCLIP-CXR's own preprocessing (open_clip defaults), identical for every model and dataset.
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

IMAGE_TOKEN = "<image>"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
# Fixed normal study from the MIMIC-CXR test set, used for the healthy-image control.
HEALTHY_MIMIC_DICOM = "c2af2ab3-6a11cbae-d9fa4d64-21ab221e-cf6f2146"
IMAGE_CONDITIONS = ("original", "black", "shuffled", "healthy", "coco", "ct")


def image_transform():
    return transforms.Compose([
        transforms.Resize(IMAGE_SIZE, interpolation=transforms.InterpolationMode.BICUBIC, antialias=True),
        transforms.CenterCrop(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def _has_indication(indication):
    return isinstance(indication, str) and indication.strip() not in ("", "None", "nan")


def build_prompt(indication=None, response=None, use_image=True, use_indication=True):
    """Vicuna v1 prompt, matching LLaVA-Rad. A response is appended only for training."""
    query = "\nProvide a description of the findings in the radiology image"
    if use_indication and _has_indication(indication):
        cleaned = indication.replace("\n", " ")
        query += f" given the following indication: {cleaned}"
    else:
        query += "."
    if use_image:
        query = IMAGE_TOKEN + query
    prompt = f"{SYSTEM_PROMPT} USER: {query} ASSISTANT:"
    if response is not None:
        prompt += f" {response}</s>"
    return prompt


def load_manifest(manifest_dir, dataset, split):
    frame = pd.read_csv(Path(manifest_dir) / f"{dataset}.csv", dtype={"patient_id": str, "study_id": str})
    frame = frame[frame["split"] == split].reset_index(drop=True)
    if frame.empty:
        raise ValueError(f"{dataset} has no rows for split {split!r}")
    return frame


def _shuffled_indices(patients, rng):
    """Permutation in which every report receives an image from a different patient."""
    patients = np.asarray(patients)
    order = rng.permutation(len(patients))
    for _ in range(100):
        clashes = np.flatnonzero(patients[order] == patients)
        if len(clashes) == 0:
            return order
        for i in clashes:
            j = rng.integers(len(patients))
            order[i], order[j] = order[j], order[i]
    raise RuntimeError("Could not assign images from different patients.")


def condition_image_paths(frame, image_dir, condition, paths, seed=0):
    """Image file for each row under an image condition. None means an all-black image."""
    if condition not in IMAGE_CONDITIONS:
        raise ValueError(f"Unknown image condition: {condition}")
    originals = [str(Path(image_dir) / image) for image in frame["image"]]
    rng = np.random.default_rng(seed)
    if condition == "original":
        return originals
    if condition == "black":
        return [None] * len(frame)
    if condition == "shuffled":
        return [originals[i] for i in _shuffled_indices(frame["patient_id"].to_numpy(), rng)]
    if condition == "healthy":
        return [str(Path(paths.image_dirs.MIMIC_CXR) / f"{HEALTHY_MIMIC_DICOM}.jpg")] * len(frame)
    pool = sorted(str(p) for p in Path(paths.control_dirs[condition]).rglob("*") if p.is_file())
    if not pool:
        raise FileNotFoundError(f"No images found for the {condition} control.")
    return [pool[i] for i in rng.integers(len(pool), size=len(frame))]


class ReportDataset(Dataset):
    def __init__(self, frame, image_paths, include_response=False, use_image=True, use_indication=True):
        self.frame = frame
        self.image_paths = image_paths
        self.include_response = include_response
        self.use_image = use_image
        self.use_indication = use_indication
        self.transform = image_transform()

    def __len__(self):
        return len(self.frame)

    def _load_image(self, path):
        image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE)) if path is None else Image.open(path).convert("RGB")
        return self.transform(image)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        findings = "" if pd.isna(row["findings"]) else str(row["findings"])
        return {
            "image": self._load_image(self.image_paths[idx]),
            "prompt": build_prompt(
                row["indication"],
                response=findings if self.include_response else None,
                use_image=self.use_image,
                use_indication=self.use_indication,
            ),
            "findings": findings,
            "patient_id": row["patient_id"],
            "study_id": row["study_id"],
            "image_id": row["image"],
        }


def collate(samples):
    batch = {key: [sample[key] for sample in samples] for key in samples[0] if key != "image"}
    batch["image"] = torch.stack([sample["image"] for sample in samples])
    return batch


class ResumableSampler(Sampler):
    """Epoch-seeded shuffling, split across ranks, that can restart partway through an epoch."""

    def __init__(self, length, batch_size, seed, rank=0, world_size=1):
        self.length = length
        self.batch_size = batch_size
        self.seed = seed
        self.rank = rank
        self.world_size = world_size
        self.epoch = 0
        self.start_batch = 0
        self.batches_per_epoch = (length // world_size) // batch_size

    def set_state(self, epoch, start_batch=0):
        self.epoch = epoch
        self.start_batch = start_batch

    def __iter__(self):
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(self.length, generator=generator).tolist()
        usable = self.batches_per_epoch * self.batch_size * self.world_size
        mine = order[:usable][self.rank :: self.world_size]
        return iter(mine[self.start_batch * self.batch_size :])

    def __len__(self):
        return (self.batches_per_epoch - self.start_batch) * self.batch_size


def make_loader(dataset, batch_size, num_workers=8, sampler=None):
    return DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate,
        pin_memory=True,
        drop_last=sampler is not None,
    )
