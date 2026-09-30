"""Measured finite-suite stress failures and reproducible audit observations.

The 98 transformations overwrite each 2x2 patch of an 8x8 digit with either
0 or 1. A failure means that the transformed image's prediction differs from
the original dataset label. This is an operational stress-test definition;
the transformations are not guaranteed to preserve the digit's semantics.
"""
from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
from scipy.stats import hypergeom
import sklearn
from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


SEED = 20260920
D = 98
M_VALUES = [100, 500, 2000]
K_VALUES = [1, 4, 16, 64, 98]
REPETITIONS = 100
OUT = Path(__file__).resolve().parent / "empirical_results"


def write_csv(path, fieldnames, records):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)


def main():
    begin = time.perf_counter()
    OUT.mkdir(exist_ok=True, parents=True)
    data = load_digits()
    x = data.data.astype(np.float64) / 16.0
    y = data.target
    train_idx, test_idx = train_test_split(
        np.arange(len(y)), test_size=0.4, stratify=y, random_state=SEED
    )
    np.savez_compressed(OUT / "dataset_split.npz", train_indices=train_idx,
                        test_indices=test_idx, test_labels=y[test_idx])
    attack_specs = [
        {"attack_id": 2 * (row * 7 + col) + value, "row": row,
         "column": col, "fill_value": value, "patch_height": 2,
         "patch_width": 2}
        for row in range(7) for col in range(7) for value in (0, 1)
    ]
    write_csv(OUT / "attack_suite.csv", list(attack_specs[0]), attack_specs)
    classifiers = {
        "logistic_regression": LogisticRegression(C=1.0, max_iter=3000),
        "random_forest": RandomForestClassifier(
            n_estimators=200, min_samples_leaf=1, random_state=SEED, n_jobs=2
        ),
    }
    report = {
        "seed": SEED, "dataset": "sklearn.datasets.load_digits",
        "dataset_shape": list(x.shape), "pixel_scaling": "divide by 16",
        "dataset_sha256": hashlib.sha256(data.data.tobytes() + y.tobytes()).hexdigest(),
        "train_size": len(train_idx), "test_size": len(test_idx),
        "split": "stratified train_test_split(test_size=0.4, random_state=20260920)",
        "suite_size": D,
        "failure_definition": "prediction after patch differs from the original dataset label",
        "eligibility": "test inputs classified correctly before patching, separately for each model",
        "interpretation": (
            "Controlled finite-suite wrong-label stress test. Patches can change digit semantics; "
            "results are not a claim of label-preserving adversarial robustness. The empirical "
            "target population is the fixed eligible test pool, sampled with replacement."
        ),
        "audit": {"m_values": M_VALUES, "k_values": K_VALUES,
                  "repetitions": REPETITIONS, "input_sampling": "independent uniform with replacement",
                  "attack_sampling": "uniform size-k subset without replacement, using the exact Hypergeom(98,S,k) law",
                  "histogram_axes": ["model", "m", "k", "repetition", "success_count_t"],
                  "histogram_padding": "t>k entries are zero",
                  "rng": "numpy.random.default_rng(SeedSequence([seed, model_index, m, k, repetition]))"},
        "environment": {"python": sys.version, "platform": platform.platform(),
                        "numpy": np.__version__, "scipy": scipy.__version__,
                        "scikit_learn": sklearn.__version__, "matplotlib": matplotlib.__version__},
        "models": {},
    }
    histograms = np.zeros((len(classifiers), len(M_VALUES), len(K_VALUES),
                           REPETITIONS, D + 1), dtype=np.int32)
    audit_rows = []
    distributions = []
    curves = []
    for model_index, (name, model) in enumerate(classifiers.items()):
        model_begin = time.perf_counter()
        print(f"Fitting {name}", flush=True)
        fit_begin = time.perf_counter()
        model.fit(x[train_idx], y[train_idx])
        fit_seconds = time.perf_counter() - fit_begin
        clean_predictions = model.predict(x[test_idx])
        eligible = clean_predictions == y[test_idx]
        eligible_test_rows = np.flatnonzero(eligible)
        eligible_idx = test_idx[eligible]
        x_eligible = x[eligible_idx]
        labels = y[eligible_idx]
        matrix = np.zeros((len(labels), D), dtype=np.uint8)
        attack_predictions = np.empty((len(labels), D), dtype=np.int8)
        evaluation_begin = time.perf_counter()
        for spec in attack_specs:
            changed = x_eligible.reshape(-1, 8, 8).copy()
            row, col = spec["row"], spec["column"]
            changed[:, row:row + 2, col:col + 2] = spec["fill_value"]
            predictions = model.predict(changed.reshape(-1, 64))
            attack_predictions[:, spec["attack_id"]] = predictions
            matrix[:, spec["attack_id"]] = predictions != labels
        evaluate_seconds = time.perf_counter() - evaluation_begin
        s = matrix.sum(axis=1).astype(np.int32)
        counts = np.bincount(s, minlength=D + 1)
        theta = float(np.mean(s > 0))
        np.savez_compressed(
            OUT / f"{name}_failure_matrix.npz", failure_matrix=matrix,
            attack_predictions=attack_predictions, eligible_dataset_indices=eligible_idx,
            eligible_test_rows=eligible_test_rows, true_labels=labels,
            s=s, s_counts=counts, all_test_indices=test_idx,
            clean_predictions=clean_predictions, clean_correct_mask=eligible,
        )
        s_for_row = dict(zip(eligible_test_rows.tolist(), s.tolist()))
        write_csv(OUT / f"{name}_test_metadata.csv",
                  ["test_row", "dataset_index", "true_label", "clean_prediction",
                   "clean_correct", "evaluated_attack_count", "wrong_label_attack_count", "any_suite_failure"],
                  [{"test_row": test_row, "dataset_index": int(dataset_idx),
                    "true_label": int(y[dataset_idx]), "clean_prediction": int(clean_predictions[test_row]),
                    "clean_correct": int(eligible[test_row]),
                    "evaluated_attack_count": D if eligible[test_row] else 0,
                    "wrong_label_attack_count": s_for_row.get(test_row, ""),
                    "any_suite_failure": int(s_for_row[test_row] > 0) if test_row in s_for_row else ""}
                   for test_row, dataset_idx in enumerate(test_idx)])
        model_report = {
            "parameters": model.get_params(),
            "clean_accuracy": float(accuracy_score(y[test_idx], clean_predictions)),
            "eligible_inputs": int(len(s)), "suite_evaluations": int(matrix.size),
            "suite_failure_theta": theta, "inputs_with_suite_failure": int(np.sum(s > 0)),
            "s_mean": float(s.mean()), "s_median": float(np.median(s)),
            "s_quantiles": {str(q): float(np.quantile(s, q)) for q in [0, .25, .5, .75, .9, .95, 1]},
            "s_counts": counts.tolist(),
            "positive_s_counts": {str(i): int(counts[i]) for i in range(1, D + 1) if counts[i]},
            "fit_seconds": fit_seconds, "suite_evaluation_seconds": evaluate_seconds,
            "detection_probability": {},
        }
        distributions.append((name, counts, len(s)))
        curve = np.array([np.mean(hypergeom.sf(0, D, s, k)) for k in range(1, D + 1)])
        curves.append((name, curve, theta))
        for k in K_VALUES:
            model_report["detection_probability"][str(k)] = float(curve[k - 1])
        for mi, m in enumerate(M_VALUES):
            for ki, k in enumerate(K_VALUES):
                for repetition in range(REPETITIONS):
                    rng = np.random.default_rng(np.random.SeedSequence([SEED, model_index, m, k, repetition]))
                    sampled_s = s[rng.integers(0, len(s), size=m)]
                    t = rng.hypergeometric(sampled_s, D - sampled_s, k)
                    h = np.bincount(t, minlength=D + 1)
                    assert h.sum() == m and np.all(h[k + 1:] == 0)
                    histograms[model_index, mi, ki, repetition] = h
                    audit_rows.append({"model": name, "m": m, "k": k,
                                       "repetition": repetition, "target_theta": theta,
                                       "observed_detection_rate": float(np.mean(t > 0)),
                                       "observed_mean_success_count": float(np.mean(t)),
                                       "draws_with_positive_s": int(np.sum(sampled_s > 0))})
        model_report["total_seconds"] = time.perf_counter() - model_begin
        report["models"][name] = model_report
        print(json.dumps({"model": name, "eligible": len(s), "clean_accuracy": model_report["clean_accuracy"],
                          "theta": theta, "s_mean": float(s.mean()), "seconds": model_report["total_seconds"]}), flush=True)
    np.savez_compressed(OUT / "audit_histograms.npz", histograms=histograms,
                        model_names=np.array(list(classifiers)), m_values=np.array(M_VALUES),
                        k_values=np.array(K_VALUES), seed=np.array(SEED), repetitions=np.array(REPETITIONS))
    write_csv(OUT / "audit_trials.csv", list(audit_rows[0]), audit_rows)
    write_csv(OUT / "s_distribution.csv", ["model", "s", "input_count", "pool_fraction"],
              [{"model": name, "s": s, "input_count": int(count), "pool_fraction": float(count / n)}
               for name, counts, n in distributions for s, count in enumerate(counts)])
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.1), sharey=True, constrained_layout=True)
    for ax, (name, counts, n), color in zip(axes, distributions, ["#1d6488", "#aa4a34"]):
        ax.bar(np.arange(D + 1), counts / n, color=color, width=0.92)
        ax.set(xlim=(-1.5, 98.5), xlabel="Number of suite attacks producing a wrong label, S",
               title=name.replace("_", " ").title())
        ax.set_xticks([0, 20, 40, 60, 80, 98])
        ax.grid(axis="y", alpha=0.2)
        ax.text(.97, .94, f"n = {n}\nP(S > 0) = {1 - counts[0] / n:.3f}",
                transform=ax.transAxes, ha="right", va="top")
    axes[0].set_ylabel("Fraction of eligible test inputs")
    fig.savefig(OUT / "empirical_s_distribution.pdf")
    fig.savefig(OUT / "empirical_s_distribution.png", dpi=220)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(5.5, 3.2), constrained_layout=True)
    for (name, curve, theta), color in zip(curves, ["#1d6488", "#aa4a34"]):
        ax.plot(np.arange(1, D + 1), curve, label=name.replace("_", " ").title(), color=color)
        ax.axhline(theta, color=color, ls=":", lw=1)
    ax.set(xlabel="Distinct suite attacks sampled per input, k",
           ylabel="Probability of observing a suite failure", xlim=(1, D), ylim=(0, 1))
    ax.legend(frameon=False, loc="lower right")
    ax.grid(alpha=0.2)
    fig.savefig(OUT / "empirical_detection_curve.pdf")
    fig.savefig(OUT / "empirical_detection_curve.png", dpi=220)
    plt.close(fig)
    write_csv(OUT / "detection_curve.csv", ["model", "k", "observed_failure_probability", "suite_failure_theta"],
              [{"model": name, "k": k, "observed_failure_probability": float(curve[k - 1]),
                "suite_failure_theta": theta} for name, curve, theta in curves for k in range(1, D + 1)])
    report["total_seconds"] = time.perf_counter() - begin
    (OUT / "empirical_summary.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Saved experiment and {len(audit_rows)} audit trials to {OUT}; elapsed {report['total_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
