"""Build the manifests from each dataset's official release. Manifests stay local: never share them.

MIMIC_CXR  LLaVA-Rad annotations from PhysioNet (llava-rad-mimic-cxr-annotations 1.0.0).
IUXRAY     OpenI ecgen-radiology XML reports plus a table of image projections.
CHEXPERT   CheXpert Plus report table. Evaluation only, sampled by patient with a fixed seed.
"""

import argparse
import glob
import json
import os
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd

COLUMNS = ["patient_id", "study_id", "image", "indication", "findings", "split"]
CHEXPERT_PATIENT_FRACTION = 0.075
CHEXPERT_SEED = 0


def prepare_mimic(annotation_dir):
    rows = []
    for split in ("train", "val", "test"):
        path = Path(annotation_dir) / f"chat_{split}_MIMIC_CXR_all_gpt4extract_rulebased_v1.json"
        for item in json.loads(path.read_text()):
            if item.get("view") == "LATERAL":
                continue
            if split == "test" and item.get("generate_method") != "rule-based":
                continue  # LLaVA-Rad evaluates on rule-based reports only.
            parts = Path(item["image"]).parts
            subject = next(part for part in parts if part.startswith("p") and len(part) == 9)[1:]
            study = next(part for part in parts if part.startswith("s"))[1:]
            rows.append({
                "patient_id": subject, "study_id": study, "image": Path(item["image"]).name,
                "indication": item.get("reason"), "findings": item["conversations"][1]["value"], "split": split,
            })
    return pd.DataFrame(rows, columns=COLUMNS)


def _parse_iu_report(path):
    root = ET.parse(path).getroot()
    sections = {node.get("Label"): (node.text or "").strip() or None for node in root.findall(".//AbstractText")}
    images = [f"{node.get('id')}.png" for node in root.findall(".//parentImage") if node.get("id")]
    return sections, images


def prepare_iuxray(report_dir, projection_csv):
    projections = pd.read_csv(projection_csv).set_index("image")["projection"].to_dict()
    rows = []
    for path in sorted(glob.glob(os.path.join(report_dir, "*.xml"))):
        report_id = Path(path).stem
        sections, images = _parse_iu_report(path)
        frontal = [image for image in images if projections.get(image) == "Frontal"]
        if not sections.get("FINDINGS") or not frontal:
            continue
        rows.append({
            "patient_id": report_id, "study_id": report_id, "image": frontal[-1],
            "indication": sections.get("INDICATION"), "findings": sections["FINDINGS"], "split": "test",
        })
    return pd.DataFrame(rows, columns=COLUMNS)


def prepare_chexpert(report_csv, image_dir):
    table = pd.read_csv(report_csv)
    table = table[(table["frontal_lateral"] == "Frontal") & table["section_findings"].fillna("").str.strip().ne("")]
    table = table.assign(image=table["path_to_image"].map(lambda p: "-".join(p.split("/")[1:])))
    available = {entry.name for entry in os.scandir(image_dir)}
    table = table[table["image"].isin(available)]

    patients = np.sort(table["deid_patient_id"].unique())
    count = round(len(patients) * CHEXPERT_PATIENT_FRACTION)
    chosen = np.random.default_rng(CHEXPERT_SEED).choice(patients, size=count, replace=False)
    table = table[table["deid_patient_id"].isin(chosen)].sort_values("path_to_image")
    return pd.DataFrame({
        "patient_id": table["deid_patient_id"],
        "study_id": table["path_to_image"].map(lambda p: p.split("/")[2]),
        "image": table["image"], "indication": None, "findings": table["section_findings"], "split": "test",
    })[COLUMNS]


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out_dir", default="manifests")
    parser.add_argument("--mimic_annotations")
    parser.add_argument("--iu_reports")
    parser.add_argument("--iu_projections")
    parser.add_argument("--chexpert_reports")
    parser.add_argument("--chexpert_images")
    args = parser.parse_args()

    manifests = {}
    if args.mimic_annotations:
        manifests["MIMIC_CXR"] = prepare_mimic(args.mimic_annotations)
    if args.iu_reports:
        manifests["IUXRAY"] = prepare_iuxray(args.iu_reports, args.iu_projections)
    if args.chexpert_reports:
        manifests["CHEXPERT"] = prepare_chexpert(args.chexpert_reports, args.chexpert_images)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, frame in manifests.items():
        frame.to_csv(out_dir / f"{name}.csv", index=False)
        sizes = frame.groupby("split").agg(rows=("image", "size"), patients=("patient_id", "nunique"))
        print(f"{name}\n{sizes.to_string()}\n")


if __name__ == "__main__":
    main()
