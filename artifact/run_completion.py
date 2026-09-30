"""Prespecified completion study on frozen measured and synthetic populations."""
from __future__ import annotations
import argparse
import csv
import json
import math
import platform
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy

from completion_audit import (allocation, completion_trial, cp_bounds,
                              detection_trial, exact_completion_moments,
                              first_hit_distribution, full_trial)
from confidence import confidence_interval, sampling_matrix
from audit_data import load_populations
from audit_inference import (fit_linear_estimator, linear_interval,
                                linear_population_risk)

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "completion_results"
SEED = 20260922
DELTA = .05
TAU = .05
PRECISION_WIDTH = .10
FRACTIONS = [.25, .50, .75]
K_VALUES = [4, 16]
THETAS = [.001, .01, .05, .2, .5, .9]
BUDGET_MULTIPLIERS = [50, 200, 800]


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value


def csv_write(path, rows):
    pd.DataFrame(rows).to_csv(path, index=False)


def design_moments(pop, first, method, B, k, fraction):
    d, theta = pop["d"], pop["theta"]
    if method == "completion":
        m, R = allocation(B, d, k, fraction)
        return exact_completion_moments(first, m, R, k)
    if method == "full":
        m = B // d
        mean_calls = m * np.dot(first, np.minimum(np.arange(1, d + 2), d))
        return dict(expected_point=theta, exact_bias=0., exact_variance=theta * (1 - theta) / m,
                    exact_mse=theta * (1 - theta) / m, expected_calls=float(mean_calls))
    if method == "linear":
        risk = linear_population_risk(pop["mu"], fit_linear_estimator(d, k, B // k))
        return dict(exact_raw_bias=risk["raw_bias"], exact_raw_variance=risk["raw_variance"],
                    exact_raw_mse=risk["raw_mse"], expected_calls=(B // k) * k)
    return dict(expected_calls=(B // k) * k) if method == "count_midpoint" else {}


def annotate(outcome, pop, study, B, method, k, fraction, rep):
    row = dict(study=study, population=pop["name"], identifier=pop["identifier"],
               family=pop["family"], synthetic=bool(pop.get("synthetic", False)),
               d=pop["d"], theta=pop["theta"], budget=B, method=method,
               k=k, screen_fraction=fraction, repetition=rep)
    row.update(outcome)
    row["effective_budget_ceiling"] = (row["m"] * k + row["R"] * (pop["d"] - k)
                                       if method == "completion" else row["m"] * (pop["d"] if method == "full" else k))
    row["error"] = row["point"] - row["theta"]
    row["squared_error"] = row["error"] ** 2
    row["raw_error"] = row["raw_point"] - row["theta"]
    row["raw_squared_error"] = row["raw_error"] ** 2
    row["width"] = row["upper"] - row["lower"]
    row["covered"] = bool(row["lower"] - 1e-10 <= row["theta"] <= row["upper"] + 1e-10)
    row["upper_covered"] = bool(row["theta"] <= row["cert_upper"] + 1e-10)
    row["certified_at_tau"] = bool(row["cert_upper"] <= TAU)
    row["width_at_most_precision"] = bool(row["width"] <= PRECISION_WIDTH)
    row["budget_used_fraction"] = row["calls"] / B
    assert row["calls"] <= B and 0 <= row["lower"] <= row["upper"] <= 1
    return row


def run_population(pop, study, B, ks, fractions, repetitions, seed_key):
    first = first_hit_distribution(pop["mu"])
    rows, moments = [], []
    d = pop["d"]
    designs = [("full", 0, 0.)]
    designs.extend(("completion", k, fraction) for k in ks for fraction in fractions)
    designs.extend(("detection_midpoint", k, 0.) for k in ks)
    designs.extend(("count_pair", k, 0.) for k in ks)
    for design_id, (method, k, fraction) in enumerate(designs):
        # This is a design feasibility rule, independent of any outcome.
        # Fractions below k/d reserve more completions than screened inputs.
        if method == "completion" and fraction < k / d:
            continue
        if method == "completion":
            m, R = allocation(B, d, k, fraction)
        elif method == "full":
            m, R = B // d, 0
        else:
            m, R = B // k, 0
        if method == "count_pair":
            q = sampling_matrix(d, k) @ pop["mu"]
            q = np.maximum(q, 0.)
            q /= q.sum()
            fit = fit_linear_estimator(d, k, m)
        for rep in range(repetitions):
            rng = np.random.default_rng(np.random.SeedSequence([SEED, *seed_key, B, design_id, rep]))
            if method == "completion":
                outcome = completion_trial(rng, first, m, R, k, DELTA)
            elif method == "full":
                outcome = full_trial(rng, first, m, DELTA)
            elif method == "detection_midpoint":
                outcome = detection_trial(rng, first, m, k, DELTA)
            else:
                hist = rng.multinomial(m, q)
                ci = confidence_interval(hist, d, k, DELTA)
                point = (ci["lower"] + ci["upper"]) / 2
                common = dict(calls=m * k, m=m, R=0, screen_negatives=int(hist[0]), completed=0,
                              completed_positive=0)
                outcome = dict(point=point, raw_point=point, lower=ci["lower"], upper=ci["upper"],
                               cert_upper=ci["upper"], solver_status=ci["status"], **common)
                rows.append(annotate(outcome, pop, study, B, "count_midpoint", k, 0., rep))
                linear = linear_interval(hist, fit, DELTA)
                upper_radius = fit["bias_bound"] + fit["range"] * np.sqrt(np.log(1 / DELTA) / (2 * m))
                outcome = dict(point=linear["point"], raw_point=linear["raw_point"],
                               lower=linear["lower"], upper=linear["upper"],
                               cert_upper=float(np.clip(linear["raw_point"] + upper_radius, 0, 1)), **common)
                rows.append(annotate(outcome, pop, study, B, "linear", k, 0., rep))
                continue
            rows.append(annotate(outcome, pop, study, B, method, k, fraction, rep))
        for output_method in (["count_midpoint", "linear"] if method == "count_pair" else [method]):
            extra = design_moments(pop, first, output_method, B, k, fraction)
            extra.pop("theta", None)
            moments.append(dict(study=study, population=pop["name"], theta=pop["theta"], budget=B,
                                method=output_method, k=k, screen_fraction=fraction, **extra))
    return rows, moments


def summarize(rows, moments):
    frame = pd.DataFrame(rows)
    keys = ["study", "population", "identifier", "family", "synthetic", "d", "theta", "budget", "method", "k", "screen_fraction"]
    summaries = []
    for key, g in frame.groupby(keys, sort=False, dropna=False):
        out = dict(zip(keys, key))
        n = len(g)
        out.update(repetitions=n, m=int(g["m"].iloc[0]), R=int(g["R"].iloc[0]))
        out["effective_budget_ceiling"] = int(g["effective_budget_ceiling"].iloc[0])
        for name, values in [("bias", g["error"]), ("mse", g["squared_error"]),
                             ("raw_bias", g["raw_error"]), ("raw_mse", g["raw_squared_error"]),
                             ("mean_width", g["width"]), ("mean_calls", g["calls"]),
                             ("mean_budget_used_fraction", g["budget_used_fraction"])]:
            out[name] = float(values.mean())
            out[name + "_mcse"] = float(values.std(ddof=1) / np.sqrt(n))
        points = g["point"].to_numpy()
        centered_sq = (points - points.mean()) ** 2
        out["variance"] = float(points.var(ddof=1))
        out["variance_mcse"] = float(np.std(centered_sq, ddof=1) / np.sqrt(n))
        out["rmse"] = float(np.sqrt(out["mse"]))
        out["rmse_mcse_delta_method"] = out["mse_mcse"] / (2 * out["rmse"]) if out["rmse"] > 0 else 0.
        for source, label in [("covered", "coverage"), ("upper_covered", "upper_coverage"),
                              ("certified_at_tau", "certification_probability"),
                              ("width_at_most_precision", "precision_probability")]:
            successes = int(g[source].sum())
            lo, hi = cp_bounds(successes, n, .05)
            out[label] = successes / n
            out[label + "_mc_lower"] = lo
            out[label + "_mc_upper"] = hi
        out["calls_max"] = int(g["calls"].max())
        out["completion_cap_reached_probability"] = float((g["completed"] == g["R"]).mean()) if out["method"] == "completion" else np.nan
        out["mean_completions"] = float(g["completed"].mean())
        out["mean_screen_negatives"] = float(g["screen_negatives"].mean())
        out["projection_solver_failures"] = int((g.get("solver_status", pd.Series(index=g.index, dtype=str)) == "uninformative_solver_failure").sum())
        summaries.append(out)
    summary = pd.DataFrame(summaries)
    moment_frame = pd.DataFrame(moments)
    return summary.merge(moment_frame, on=["study", "population", "theta", "budget", "method", "k", "screen_fraction"], how="left")


def validation(populations):
    # Direct enumeration of uniform attack permutations for a tiny suite tests
    # the first-hit sampler independently of its probability recurrence.
    import itertools
    d = 5
    worst = 0.
    for s in range(d + 1):
        mu = np.eye(d + 1)[s]
        first = first_hit_distribution(mu)
        observed = np.zeros(d + 1)
        for order in itertools.permutations(range(d)):
            hit = next((j for j, attack in enumerate(order) if attack < s), d)
            observed[hit] += 1 / math.factorial(d)
        worst = max(worst, float(np.max(np.abs(first - observed))))
    assert worst < 1e-12
    # A realized all-completed sample must retain simultaneous CP unless the
    # design guaranteed full observation. Check a known distinct case.
    from completion_audit import completion_interval
    simultaneous = completion_interval(8, 10, 1, 2, 3)
    full = cp_bounds(9, 10, .05)
    assert not np.allclose(simultaneous, full)
    deterministic = completion_interval(8, 10, 1, 2, 10)
    assert np.allclose(deterministic, full)
    # Monte Carlo cross-check at a deliberately nondegenerate synthetic point.
    pop = next(p for p in populations if p["name"] == "Forest")
    mu = pop["mu"].copy()
    mu[1:] *= .2 / mu[1:].sum()
    mu[0] = .8
    first = first_hit_distribution(mu)
    m, R, k = 120, 12, 4
    exact = exact_completion_moments(first, m, R, k)
    n = 40000
    rng = np.random.default_rng(9172301)
    points, costs = np.empty(n), np.empty(n)
    for i in range(n):
        trial = completion_trial(rng, first, m, R, k)
        points[i], costs[i] = trial["point"], trial["calls"]
    mean_se = np.std(points, ddof=1) / np.sqrt(n)
    centered_sq = (points - exact["theta"]) ** 2
    mse_se = np.std(centered_sq, ddof=1) / np.sqrt(n)
    cost_se = np.std(costs, ddof=1) / np.sqrt(n)
    mean_z = (points.mean() - exact["theta"]) / mean_se
    mse_z = (centered_sq.mean() - exact["exact_variance"]) / mse_se
    cost_z = (costs.mean() - exact["expected_calls"]) / cost_se
    assert max(abs(mean_z), abs(mse_z), abs(cost_z)) < 5
    return dict(first_hit_permutation_max_residual=worst,
                outcome_dependent_full_cp_reduction="correctly_disabled",
                deterministic_full_cp_reduction="passed", variance_crosscheck_repetitions=n,
                variance_crosscheck_design=dict(m=m, R=R, k=k, theta=.2),
                exact_moments=exact, simulated_mean=float(points.mean()),
                simulated_variance=float(points.var(ddof=1)), simulated_mse=float(centered_sq.mean()),
                simulated_mean_calls=float(costs.mean()), mean_z=float(mean_z), mse_z=float(mse_z), cost_z=float(cost_z))


def make_figures(main, controlled):
    plt.rcParams.update({"font.size": 11.5, "axes.spines.top": False, "axes.spines.right": False,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.5), constrained_layout=True)
    for ax, (name, g) in zip(axes.flat, main.groupby("population", sort=False)):
        for k, color in [(4, "#246c97"), (16, "#b45133")]:
            sub = g[(g.method == "completion") & (g.k == k)].sort_values("screen_fraction")
            ax.plot(sub.mean_budget_used_fraction, sub.mean_width, color=color, lw=.8)
            for _, row in sub.iterrows():
                marker = {.25: "o", .5: "s", .75: "^"}[row.screen_fraction]
                ax.errorbar(row.mean_budget_used_fraction, row.mean_width,
                            yerr=1.96 * row.mean_width_mcse, marker=marker, color=color,
                            markersize=5, capsize=2, linestyle="none")
        full = g[g.method == "full"]
        ax.scatter(full.mean_budget_used_fraction, full.mean_width,
                   marker="*", color="#111111", s=48)
        baseline_styles = [("count_midpoint", "D", "D", "#777777"),
                           ("linear", "x", "X", "#7c4a99"),
                           ("detection_midpoint", "+", "P", "#267b53")]
        for method, marker4, marker16, color in baseline_styles:
            for k, marker in [(4, marker4), (16, marker16)]:
                sub = g[(g.method == method) & (g.k == k)]
                if k == 4:
                    ax.scatter(sub.mean_budget_used_fraction, sub.mean_width,
                               marker=marker, color=color, s=29, zorder=4)
                else:
                    ax.scatter(sub.mean_budget_used_fraction, sub.mean_width,
                               marker=marker, facecolors="white", edgecolors=color,
                               linewidths=1.2, s=37, zorder=5)
        # At x=1 the Logistic/Forest baselines nearly coincide.  Callouts
        # expose every point without shifting its measured coordinates.
        if name in ["Logistic", "Forest"]:
            for method, color, side in [("count_midpoint", "#777777", -1),
                                         ("linear", "#7c4a99", 1)]:
                sub = g[g.method == method].sort_values("mean_width")
                for i, (_, row) in enumerate(sub.iterrows()):
                    label_position = (.94, .43 if row.k == 4 else .29) if side < 0 else (10, -8 if i == 0 else 8)
                    ax.annotate(f"k={row.k}",
                                xy=(row.mean_budget_used_fraction, row.mean_width),
                                xytext=label_position, textcoords="data" if side < 0 else "offset points",
                                ha="right" if side < 0 else "left", va="center",
                                fontsize=8.5, color=color,
                                arrowprops=dict(arrowstyle="-", color=color, lw=.6),
                                annotation_clip=False)
        ax.set(title=name, xlabel="Mean attack evaluations / cap", ylabel="Mean 95% interval width", xlim=(0, 1.22), yscale="log", ylim=(.001, 1.1))
        ax.set_xticks([0, .25, .5, .75, 1.0])
        ax.grid(alpha=.16)
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color="#246c97", label="Completion k=4"),
               Line2D([], [], color="#b45133", label="Completion k=16"),
               Line2D([], [], color="#111111", marker="*", ls="", label="Full"),
               Line2D([], [], color="black", marker="o", ls="", label="Screen fraction .25"),
               Line2D([], [], color="black", marker="s", ls="", label="Screen fraction .50"),
               Line2D([], [], color="black", marker="^", ls="", label="Screen fraction .75"),
               Line2D([], [], color="#777777", marker="D", ls="", label="Count projection k=4"),
               Line2D([], [], color="#7c4a99", marker="x", ls="", label="Linear k=4"),
               Line2D([], [], color="#267b53", marker="+", ls="", label="Detection k=4"),
               Line2D([], [], color="#777777", marker="D", markerfacecolor="white", ls="", label="Count projection k=16"),
               Line2D([], [], color="#7c4a99", marker="X", markerfacecolor="white", ls="", label="Linear k=16"),
               Line2D([], [], color="#267b53", marker="P", markerfacecolor="white", ls="", label="Detection k=16")]
    fig.legend(handles=handles, loc="outside lower center", ncol=4, fontsize=9.5,
               columnspacing=1.6, handletextpad=.7)
    fig.savefig(OUT / "main_cost_width.pdf")
    fig.savefig(OUT / "main_cost_width.png", dpi=190)
    plt.close(fig)
    fig, axes = plt.subplots(2, 3, figsize=(10.2, 6.2), constrained_layout=True, sharex=True, sharey=True)
    colors = {"completion": "#246c97", "full": "#111111", "count_midpoint": "#777777",
              "linear": "#7c4a99", "detection_midpoint": "#267b53"}
    labels = {"completion": "Completion", "full": "Full", "count_midpoint": "Count projection",
              "linear": "Linear", "detection_midpoint": "Detection"}
    for ri, (name, g) in enumerate(controlled.groupby("population", sort=False)):
        for ci, B in enumerate(sorted(g.budget.unique())):
            ax = axes[ri, ci]
            for method, sub in g[g.budget == B].groupby("method", sort=False):
                sub = sub.sort_values("theta")
                y = sub.certification_probability.to_numpy()
                err = np.vstack([y - sub.certification_probability_mc_lower.to_numpy(),
                                 sub.certification_probability_mc_upper.to_numpy() - y])
                ax.errorbar(sub.theta, y, yerr=err, color=colors[method], marker="o",
                            markersize=3, capsize=2, lw=1, label=labels[method])
            ax.axvline(TAU, ls=":", color="#aa2222", lw=1)
            ax.set(xscale="log", title=f"{name}; B = {B:,}", xlabel="Synthetic suite-failure probability", ylabel="P(95% upper bound ≤ .05)", ylim=(-.03, 1.03))
            ax.grid(alpha=.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncol=3, fontsize=10.5)
    fig.savefig(OUT / "controlled_certification.pdf")
    fig.savefig(OUT / "controlled_certification.png", dpi=190)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--skip-validation", action="store_true")
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True, parents=True)
    started = time.perf_counter()
    populations = load_populations()
    plan = dict(seed=SEED, repetitions=args.repetitions, main_k=K_VALUES,
                main_screen_budget_fractions=FRACTIONS, main_budget={36: 18000, 98: 19600},
                completion_design_feasibility="retain fractions f>=k/d; clip R at m",
                excluded_main_designs="StrongREJECT (d=36), k=16, f=.25 for all four populations",
                controlled_templates=["GPT-4o-mini", "Forest"], controlled_theta=THETAS,
                controlled_budget_multipliers=BUDGET_MULTIPLIERS,
                controlled_k=4, controlled_screen_budget_fraction=.5,
                certification_threshold=TAU, precision_width=PRECISION_WIDTH, delta=DELTA,
                tested_budget_attainment="lower 95% Monte Carlo confidence limit for success probability >=.90",
                no_budget_reinvestment=True,
                sampling="iid input draws with replacement; independent uniform attack orders",
                first_hit_simulation="exact mixture of order statistics from measured S distributions",
                intervals="simultaneous CP completion; CP full; support envelope detection; existing projected CP count; bias/Hoeffding linear",
                point_estimators="unbiased completion and full; clipped regularized linear; interval midpoint for detection and count projection",
                count_upper_bound="upper endpoint of two-sided 95% projected interval is a conservative 95% upper bound",
                linear_upper_bound="bias plus one-sided Hoeffding radius at delta=.05",
                data_queries="zero new model calls; all attack-call costs are replayed counts under the frozen outcomes")
    (OUT / "study_design.json").write_text(json.dumps(serializable(plan), indent=2) + "\n")
    main_rows, main_moments = [], []
    for index, pop in enumerate(populations):
        B = 18000 if pop["d"] == 36 else 19600
        rows, moments = run_population(pop, "measured", B, K_VALUES, FRACTIONS, args.repetitions, [0, index])
        main_rows.extend(rows)
        main_moments.extend(moments)
        print(f"Main {pop['name']}: {len(rows)} trial rows; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    main_summary = summarize(main_rows, main_moments)
    csv_write(OUT / "main_trials.csv", main_rows)
    main_summary.to_csv(OUT / "main_summary.csv", index=False)
    controlled_rows, controlled_moments = [], []
    for template_id, name in enumerate(plan["controlled_templates"]):
        template = next(p for p in populations if p["name"] == name)
        for theta_id, theta in enumerate(THETAS):
            mu = template["mu"].copy()
            mu[1:] *= theta / mu[1:].sum()
            mu[0] = 1 - theta
            pop = dict(template, name=name, mu=mu, theta=theta, synthetic=True)
            for B in [template["d"] * n for n in BUDGET_MULTIPLIERS]:
                rows, moments = run_population(pop, "controlled", B, [4], [.5], args.repetitions,
                                               [1, template_id, theta_id])
                controlled_rows.extend(rows)
                controlled_moments.extend(moments)
            print(f"Controlled {name} theta={theta:g}; elapsed {time.perf_counter()-started:.1f}s", flush=True)
    controlled_summary = summarize(controlled_rows, controlled_moments)
    csv_write(OUT / "controlled_trials.csv", controlled_rows)
    controlled_summary.to_csv(OUT / "controlled_summary.csv", index=False)
    # Smallest TESTED budget attaining the prespecified precision/certification
    # success rate. This does not interpolate or claim a minimal budget.
    budget_rows = []
    for key, g in controlled_summary.groupby(["population", "theta", "method"], sort=False):
        g = g.sort_values("budget")
        for criterion, column in [("width_at_most_.10", "precision_probability"),
                                  ("upper_at_most_.05", "certification_probability")]:
            passing = g[g[column + "_mc_lower"] >= .90]
            budget_rows.append(dict(population=key[0], theta=key[1], method=key[2], criterion=criterion,
                                    smallest_tested_budget_with_mc_lower_success_rate_at_least_90pct=int(passing.budget.iloc[0]) if len(passing) else np.nan,
                                    attained=bool(len(passing))))
    csv_write(OUT / "tested_budget_thresholds.csv", budget_rows)
    if not args.skip_validation:
        check = validation(populations)
        (OUT / "validation.json").write_text(json.dumps(serializable(check), indent=2) + "\n")
    make_figures(main_summary, controlled_summary)
    metadata = dict(plan, elapsed_seconds=time.perf_counter()-started, python=sys.version,
                    numpy=np.__version__, scipy=scipy.__version__, pandas=pd.__version__,
                    platform=platform.platform(), measured_trial_rows=len(main_rows),
                    controlled_trial_rows=len(controlled_rows), main_summary_rows=len(main_summary),
                    controlled_summary_rows=len(controlled_summary))
    (OUT / "run_metadata.json").write_text(json.dumps(serializable(metadata), indent=2) + "\n")
    print(f"Finished in {metadata['elapsed_seconds']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
