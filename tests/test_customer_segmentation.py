"""Tests for src/models/customer_segmentation.py."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

from src.models import customer_segmentation as S


def test_log_rfm_clips_negative_spend_and_logs():
    X = pd.DataFrame({"recency_days": [0, 9], "frequency": [1, 99], "monetary": [-50.0, np.e - 1]})
    out = S.log_rfm(X)
    assert out[0, 2] == 0                                  # negative spend -> 0 -> log1p(0)
    assert out[1, 2] == pytest.approx(1.0)                 # log1p(e - 1) = 1
    assert out[1, 1] == pytest.approx(np.log(100))


def test_preprocessor_standardises():
    rng = np.random.default_rng(0)
    X = pd.DataFrame({"recency_days": rng.integers(0, 365, 500), "frequency": rng.integers(1, 50, 500),
                      "monetary": rng.lognormal(6, 1.5, 500)})
    Z = S.make_preprocessor().fit_transform(X)
    assert np.allclose(Z.mean(axis=0), 0, atol=1e-9)
    assert np.allclose(Z.std(axis=0), 1, atol=1e-9)


def _k_row(k, sil, ari):
    return {"k": k, "silhouette": sil, "stability_ari": ari}


def test_choose_k_prefers_most_detailed_stable_and_separated():
    t = pd.DataFrame([_k_row(2, 0.45, 0.99), _k_row(3, 0.34, 0.94), _k_row(4, 0.335, 0.92),
                      _k_row(5, 0.33, 0.80), _k_row(6, 0.25, 0.95)])
    k, reason = S.choose_k(t)
    assert k == 4                       # 5 unstable, 6 too poorly separated, 2 excluded
    assert reason.startswith("k = 4")


def test_choose_k_falls_back_to_most_stable():
    t = pd.DataFrame([_k_row(3, 0.30, 0.70), _k_row(4, 0.29, 0.85)])
    k, reason = S.choose_k(t)
    assert k == 4 and "most stable" in reason


def test_name_clusters_maps_shapes_to_business_names():
    centers = pd.DataFrame({"R": [1.3, -0.1, 0.8, -0.9, -0.8],
                            "F": [1.7, 0.4, -0.4, -0.8, 0.6],
                            "M": [1.4, 0.5, -0.4, -0.8, 0.3]})
    names = S.name_clusters(centers)
    assert names == {0: "Champions", 1: "Loyal regulars", 2: "New customers", 3: "Hibernating", 4: "At risk"}


def test_name_clusters_deduplicates():
    centers = pd.DataFrame({"R": [2.0, 0.1, 0.2, -2.0], "F": [2.0, 0.3, 0.2, -2.0], "M": [2.0, 0.1, 0.5, -2.0]})
    names = S.name_clusters(centers)
    assert len(set(names.values())) == 4
    assert names[2] == "Loyal regulars (higher spend)" and names[1] == "Loyal regulars (lower spend)"


def test_lapsing_flag_only_for_valuable_segments():
    df = pd.DataFrame({"segment": ["Champions", "Loyal regulars (higher spend)", "Hibernating", "Champions"],
                       "recency_days": [100, 95, 300, 10]})
    assert S.add_lapsing_flag(df).tolist() == [True, True, False, False]


def test_rule_based_segments():
    df = pd.DataFrame({"R_score": [5, 3, 5, 3, 1, 1], "F_score": [5, 3, 1, 1, 4, 1]})
    assert S.rule_based_segments(df).tolist() == ["Champions", "Loyal", "New / promising", "Need attention",
                                                  "At risk", "Hibernating"]


def test_experiment_recovers_planted_segments():
    """Three well-separated synthetic customer groups must be found again (high ARI) and be stable."""
    rng = np.random.default_rng(7)
    groups = [(5, 30, 5000), (60, 4, 800), (250, 1, 150)]          # (recency, frequency, spend)
    rows, truth = [], []
    for g, (r, f, m) in enumerate(groups):
        n = 150
        rows.append(pd.DataFrame({
            "recency_days": np.clip(rng.normal(r, r * 0.2 + 1, n), 0, 365).round(),
            "frequency": np.clip(rng.normal(f, f * 0.2 + 0.3, n), 1, None).round(),
            "monetary": rng.lognormal(np.log(m), 0.2, n)}))
        truth += [g] * n
    customers = pd.concat(rows, ignore_index=True)
    customers["avg_order_value"] = customers["monetary"] / customers["frequency"]
    customers["return_rate"] = 0.0
    customers["is_uk"] = 1
    customers["R_score"] = pd.qcut(customers["recency_days"].rank(method="first"), 5, labels=[5, 4, 3, 2, 1]).astype(int)
    customers["F_score"] = pd.qcut(customers["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)

    result = S.run_experiment(customers, k_range=range(2, 6), n_boot=5)
    assert result["k"] == 3
    assert adjusted_rand_score(truth, result["customers"]["cluster"]) > 0.95
    champions = result["customers"].loc[result["customers"]["segment"] == "Champions"]
    assert champions["recency_days"].median() < 20
