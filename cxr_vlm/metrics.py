"""Per-report scoring and paired, patient-clustered bootstrap comparison.

Each report is scored once, then resampled with one set of patient draws across all runs, so
every delta is a paired difference on the same reports. Blank targets cannot be scored and raise.
"""

import hashlib
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

# Order returned by F1CheXbert.get_label
CHEXBERT_LABELS = (
    "Enlarged Cardiomediastinum", "Cardiomegaly", "Lung Opacity", "Lung Lesion", "Edema",
    "Consolidation", "Pneumonia", "Atelectasis", "Pneumothorax", "Pleural Effusion",
    "Pleural Other", "Fracture", "Support Devices", "No Finding",
)
FIVE_LABELS = np.asarray([1, 4, 5, 7, 9])  # Cardiomegaly, Edema, Consolidation, Atelectasis, Effusion
MEAN_METRICS = ("radgraph", "rouge1", "rouge2", "rougeL")


def load_predictions(csv_path):
    frame = pd.read_csv(csv_path, dtype={"patient_id": str, "study_id": str})
    frame["prediction"] = frame["prediction"].fillna("").astype(str)
    frame["target"] = frame["target"].fillna("").astype(str)
    if frame["target"].str.strip().eq("").any():
        raise ValueError(f"{csv_path} has blank targets.")
    return frame


def _ngram_counts(tokens, max_order):
    counts = Counter()
    for order in range(1, max_order + 1):
        for start in range(len(tokens) - order + 1):
            counts[tuple(tokens[start : start + order])] += 1
    return counts


def _bleu_statistics(predictions, references):
    """Additive sufficient statistics for corpus BLEU, with the Hugging Face tokenizer."""
    import evaluate

    tokenizer = evaluate.load("bleu")._compute.__globals__["Tokenizer13a"]()
    n = len(predictions)
    matches, possible = np.zeros((n, 4), dtype=np.int64), np.zeros((n, 4), dtype=np.int64)
    translation, reference = np.zeros(n, dtype=np.int64), np.zeros(n, dtype=np.int64)
    for i, (hypothesis, target) in enumerate(zip(predictions, references)):
        hypothesis_tokens, target_tokens = tokenizer(hypothesis), tokenizer(target)
        translation[i], reference[i] = len(hypothesis_tokens), len(target_tokens)
        overlap = _ngram_counts(hypothesis_tokens, 4) & _ngram_counts(target_tokens, 4)
        for ngram, count in overlap.items():
            matches[i, len(ngram) - 1] += count
        for order in range(1, 5):
            possible[i, order - 1] = max(len(hypothesis_tokens) - order + 1, 0)
    return {"bleu_matches": matches, "bleu_possible": possible,
            "bleu_translation_length": translation, "bleu_reference_length": reference}


def _chexbert_labels(predictions, references):
    import torch
    from f1chexbert import F1CheXbert

    scorer = F1CheXbert()
    with torch.inference_mode():
        predicted = np.asarray([scorer.get_label(text.strip()) for text in predictions], dtype=np.int8)
        target = np.asarray([scorer.get_label(text.strip()) for text in references], dtype=np.int8)
    return predicted, target


def _rouge(predictions, references):
    import evaluate

    values = evaluate.load("rouge").compute(predictions=predictions, references=references, use_aggregator=False)
    return {name: np.asarray(values[name], dtype=np.float64) for name in ("rouge1", "rouge2", "rougeL")}


def _radgraph(predictions, references):
    from radgraph import F1RadGraph

    scorer = F1RadGraph(reward_level="partial", model_type="radgraph-xl")
    _, rewards, _, _ = scorer(hyps=predictions, refs=references)
    rewards = np.asarray(rewards, dtype=np.float64)
    if rewards.shape != (len(predictions),):
        raise RuntimeError(f"Unexpected RadGraph reward shape: {rewards.shape}")
    return rewards


def _reports_hash(predictions, references):
    digest = hashlib.sha256()
    for hypothesis, target in zip(predictions, references):
        digest.update(hypothesis.encode("utf-8", "surrogatepass") + b"\x00")
        digest.update(target.encode("utf-8", "surrogatepass") + b"\x01")
    return digest.hexdigest()


def score_reports(frame, cache_path):
    """Per-report statistics for one predictions file, cached next to the results."""
    predictions, references = frame["prediction"].tolist(), frame["target"].tolist()
    report_hash = _reports_hash(predictions, references)
    cache_path = Path(cache_path)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as stored:
            if str(stored["report_hash"].item()) == report_hash:
                return {key: stored[key] for key in stored.files if key != "report_hash"}
    cache = _bleu_statistics(predictions, references)
    cache["chexbert_pred_labels"], cache["chexbert_ref_labels"] = _chexbert_labels(predictions, references)
    cache.update(_rouge(predictions, references))
    cache["radgraph"] = _radgraph(predictions, references)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, report_hash=np.asarray(report_hash), **cache)
    return cache


def _f1(tp, fp, fn):
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros_like(denominator, dtype=float), where=denominator > 0)


def _bleu(cache, weights, max_order):
    matches = weights @ cache["bleu_matches"][:, :max_order]
    possible = weights @ cache["bleu_possible"][:, :max_order]
    if np.any(matches <= 0) or np.any(possible <= 0):
        return 0.0
    geometric_mean = float(np.exp(np.mean(np.log(matches / possible))))
    translation = weights @ cache["bleu_translation_length"]
    reference = weights @ cache["bleu_reference_length"]
    if translation == 0:
        return 0.0
    brevity = 1.0 if translation > reference else float(np.exp(1.0 - reference / translation))
    return geometric_mean * brevity


def weighted_metrics(cache, weights):
    """All metrics for per-report weights, on a 0-100 scale."""
    predicted = cache["chexbert_pred_labels"].astype(bool)
    target = cache["chexbert_ref_labels"].astype(bool)
    tp = weights @ (predicted & target)
    fp = weights @ (predicted & ~target)
    fn = weights @ (~predicted & target)
    per_label = _f1(tp, fp, fn)
    result = {
        "chexbert_micro_14": float(_f1(tp.sum(), fp.sum(), fn.sum())),
        "chexbert_macro_14": float(per_label.mean()),
        "chexbert_micro_5": float(_f1(tp[FIVE_LABELS].sum(), fp[FIVE_LABELS].sum(), fn[FIVE_LABELS].sum())),
        "chexbert_macro_5": float(per_label[FIVE_LABELS].mean()),
        "bleu_1": _bleu(cache, weights, 1),
        "bleu_4": _bleu(cache, weights, 4),
    }
    for name in MEAN_METRICS:
        result[name] = float(weights @ cache[name] / weights.sum())
    for label, value in zip(CHEXBERT_LABELS, per_label):
        result[f"f1_{label}"] = float(value)
    return {name: value * 100.0 for name, value in result.items()}


def paired_bootstrap(groups, patients, baseline, n_bootstrap=5000, confidence=0.95, seed=42, margin=2.0):
    """Compare groups of runs. A group's value is the mean over its runs (training or sampling seeds).

    groups maps a name to a list of per-report caches, all on the same aligned reports.
    """
    unique, patient_index = np.unique(np.asarray(patients), return_inverse=True)
    n_patients = len(unique)
    rng = np.random.default_rng(seed)

    def group_values(weights):
        return {name: [weighted_metrics(cache, weights) for cache in members] for name, members in groups.items()}

    point = group_values(np.ones(len(patients)))
    metrics = list(next(iter(point.values()))[0])
    draws = {name: np.empty((n_bootstrap, len(metrics))) for name in groups}
    for b in range(n_bootstrap):
        counts = np.bincount(rng.integers(0, n_patients, n_patients), minlength=n_patients)
        values = group_values(counts[patient_index].astype(float))
        for name, members in values.items():
            draws[name][b] = [np.mean([member[metric] for member in members]) for metric in metrics]

    alpha = (1.0 - confidence) / 2.0
    rows = []
    for name, members in point.items():
        for m, metric in enumerate(metrics):
            per_run = np.asarray([member[metric] for member in members])
            low, high = np.quantile(draws[name][:, m], [alpha, 1.0 - alpha])
            row = {"group": name, "metric": metric, "n_runs": len(members), "estimate": per_run.mean(),
                   "between_run_sd": per_run.std(ddof=1) if len(members) > 1 else np.nan,
                   "ci_lower": low, "ci_upper": high}
            if name != baseline:
                delta_low, delta_high = np.quantile(draws[name][:, m] - draws[baseline][:, m], [alpha, 1.0 - alpha])
                baseline_estimate = np.mean([member[metric] for member in point[baseline]])
                row.update({
                    "delta": per_run.mean() - baseline_estimate, "delta_lower": delta_low, "delta_upper": delta_high,
                    "non_inferior": bool(delta_low > -margin),
                    "equivalent": bool(delta_low > -margin and delta_high < margin),
                })
            rows.append(row)
    return pd.DataFrame(rows), n_patients
