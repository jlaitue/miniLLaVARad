import numpy as np
import pandas as pd
import pytest
import torch

from cxr_vlm.data import ResumableSampler, _shuffled_indices, build_prompt
from cxr_vlm.metrics import paired_bootstrap, weighted_metrics
from cxr_vlm.model import pool_tokens


def test_prompt_matches_llavarad_template():
    assert build_prompt("Cough.\nFever") == (
        "A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, "
        "detailed, and polite answers to the user's questions. USER: <image>\nProvide a description of the findings "
        "in the radiology image given the following indication: Cough. Fever ASSISTANT:"
    )
    assert build_prompt("Cough", response="Clear.").endswith("ASSISTANT: Clear.</s>")


@pytest.mark.parametrize("missing", [None, float("nan"), "None", "", "  "])
def test_missing_indication_is_not_written_into_the_prompt(missing):
    prompt = build_prompt(missing)
    assert "indication" not in prompt and "None" not in prompt and "nan" not in prompt
    assert "radiology image. ASSISTANT:" in prompt


def test_prompt_options():
    assert "<image>" not in build_prompt("Cough", use_image=False)
    assert "Cough" not in build_prompt("Cough", use_indication=False)


def test_sampling_keeps_exact_count_per_image():
    features = torch.zeros(4, 10, 3)
    for i in range(10):
        features[:, i] = float(2**i)
    pooled = pool_tokens(features, 1, sample_tokens=3, generator=torch.Generator().manual_seed(0))
    assert pooled.shape == (4, 1, 3)
    counts = [bin(int(round(value.item() * 3))).count("1") for value in pooled[:, 0, 0]]
    assert counts == [3, 3, 3, 3]
    assert pooled[:, 0, 0].unique().numel() > 1  # Each image gets its own subset.


def test_sampling_is_seeded_and_independent_of_batch_size():
    features = torch.randn(8, 100, 4)
    together = pool_tokens(features, 1, 5, torch.Generator().manual_seed(7))
    generator = torch.Generator().manual_seed(7)
    split = torch.cat([pool_tokens(features[:3], 1, 5, generator), pool_tokens(features[3:], 1, 5, generator)])
    assert torch.equal(together, split)


def test_pooling_modes():
    features = torch.randn(2, 1369, 4)
    assert torch.equal(pool_tokens(features, 1369), features)
    assert torch.equal(pool_tokens(features, 1), features.mean(1, keepdim=True))
    assert torch.equal(pool_tokens(features, 1, 1369), features.mean(1, keepdim=True))
    with pytest.raises(ValueError):
        pool_tokens(features, 1369, 5)
    with pytest.raises(ValueError):
        pool_tokens(features, 1, 0)


def test_shuffled_images_come_from_other_patients():
    patients = np.array(["a", "a", "a", "b", "b", "c", "d", "d"])
    order = _shuffled_indices(patients, np.random.default_rng(0))
    assert sorted(order) == list(range(8))
    assert (patients[order] != patients).all()


def test_sampler_splits_ranks_without_overlap_and_resumes():
    samplers = [ResumableSampler(103, 4, 0, rank, 2) for rank in range(2)]
    first, second = [list(sampler) for sampler in samplers]
    assert len(first) == len(second) == samplers[0].batches_per_epoch * 4
    assert not set(first) & set(second)

    samplers[0].set_state(0, start_batch=3)
    assert list(samplers[0]) == first[12:]
    assert len(samplers[0]) == len(first) - 12
    samplers[0].set_state(1)
    assert list(samplers[0]) != first


def fake_cache(rng, target, n, flip):
    matches = rng.integers(1, 30, (n, 4))
    possible = matches + rng.integers(0, 30, (n, 4))
    return {
        "chexbert_pred_labels": (target ^ (rng.random((n, 14)) < flip)).astype(np.int8),
        "chexbert_ref_labels": target,
        "bleu_matches": matches, "bleu_possible": possible, "bleu_translation_length": possible[:, 0],
        "bleu_reference_length": rng.integers(1, 70, n), "radgraph": rng.random(n), "rouge1": rng.random(n),
        "rouge2": rng.random(n), "rougeL": rng.random(n),
    }


def test_metrics_match_sklearn_and_row_repetition():
    from sklearn.metrics import f1_score

    rng = np.random.default_rng(0)
    n = 500
    target = (rng.random((n, 14)) < 0.2).astype(np.int8)
    cache = fake_cache(rng, target, n, 0.1)
    result = weighted_metrics(cache, np.ones(n))
    predicted = cache["chexbert_pred_labels"]
    assert result["chexbert_micro_14"] == pytest.approx(100 * f1_score(target, predicted, average="micro"))
    assert result["chexbert_macro_14"] == pytest.approx(100 * f1_score(target, predicted, average="macro"))

    weights = rng.integers(0, 3, n)
    rows = np.repeat(np.arange(n), weights)
    resampled = {key: value[rows] for key, value in cache.items()}
    assert weighted_metrics(cache, weights.astype(float)) == pytest.approx(
        weighted_metrics(resampled, np.ones(len(rows)))
    )


def test_paired_bootstrap_delta_is_zero_for_identical_runs():
    rng = np.random.default_rng(1)
    n = 300
    target = (rng.random((n, 14)) < 0.2).astype(np.int8)
    a, b = fake_cache(rng, target, n, 0.1), fake_cache(rng, target, n, 0.3)
    patients = rng.integers(0, 40, n).astype(str)
    summary, n_patients = paired_bootstrap({"base": [a], "same": [a], "worse": [a, b]}, patients, "base", n_bootstrap=100)
    assert n_patients == len(set(patients))
    same = summary[(summary.group == "same") & (summary.metric == "chexbert_micro_14")].iloc[0]
    assert same.delta == same.delta_lower == same.delta_upper == 0
    assert same.equivalent
    worse = summary[(summary.group == "worse") & (summary.metric == "chexbert_micro_14")].iloc[0]
    assert worse.delta < 0 and worse.n_runs == 2 and worse.between_run_sd > 0
