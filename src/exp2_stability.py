"""Experiment 2 - Stability: how much do explanations change with the random seed, sample budget,
LIME kernel width and SHAP background data?

Interventional TreeSHAP (deterministic) serves as the noise-free reference for the Random Forest.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap

from common import (
    BACKGROUND_SIZE,
    KERNEL_SHAP_NSAMPLES,
    LIME_NUM_SAMPLES,
    SEED,
    explain_kernel_shap,
    explain_lime,
    explain_tree,
    fit_random_forest,
    load_split,
    make_lime_explainer,
    make_tree_explainers,
    malignant_probability,
    sample_background,
    save_figure,
    save_table,
    select_instances,
    stability_summary,
    timed,
    top_k,
)

LIME_SEEDS = range(30)
LIME_NUM_SAMPLES_GRID = [500, 1000, LIME_NUM_SAMPLES]
LIME_KERNEL_WIDTH_FACTORS = [0.25, 0.75, 3.0]
KERNEL_SEEDS = range(15)
KERNEL_NSAMPLES_GRID = [100, 500, KERNEL_SHAP_NSAMPLES]


def lime_sweep(fn, X_train, x_row, case):
    """LIME on the RF over seeds x num_samples (default width) and seeds x kernel width (default samples)."""
    d = X_train.shape[1]
    configs = [(n, 0.75) for n in LIME_NUM_SAMPLES_GRID] + [
        (LIME_NUM_SAMPLES, w) for w in LIME_KERNEL_WIDTH_FACTORS if w != 0.75
    ]
    rows, runs = [], {}
    for num_samples, width_factor in configs:
        phis, seconds = [], []
        for seed in LIME_SEEDS:
            explainer = make_lime_explainer(X_train, seed=seed, kernel_width=width_factor * np.sqrt(d))
            result, elapsed = timed(explain_lime, explainer, fn, x_row, num_samples)
            phis.append(result["phi"])
            seconds.append(elapsed)
        runs[(num_samples, width_factor)] = phis
        rows.append(
            {"instance": case, "method": "LIME", "num_samples": num_samples, "kernel_width": f"{width_factor}*sqrt(d)",
             "runs": len(phis), **stability_summary(phis), "runtime_s": np.mean(seconds)}
        )
    return rows, runs


def kernel_shap_sweep(explainer, x_row, case, reference=None):
    rows, runs = [], {}
    for nsamples in KERNEL_NSAMPLES_GRID:
        phis, seconds = [], []
        for seed in KERNEL_SEEDS:
            phi, elapsed = timed(explain_kernel_shap, explainer, x_row, nsamples, seed)
            phis.append(phi)
            seconds.append(elapsed)
        runs[nsamples] = phis
        error = np.nan if reference is None else float(np.mean([(phi - reference).abs().mean() for phi in phis]))
        rows.append(
            {"instance": case, "method": "Kernel SHAP", "num_samples": nsamples, "kernel_width": "-",
             "runs": len(phis), **stability_summary(phis), "mean_abs_error_vs_reference": error,
             "runtime_s": np.mean(seconds)}
        )
    return rows, runs


def background_sets(X_train, y_train) -> dict:
    benign, malignant = X_train[y_train == 0], X_train[y_train == 1]
    return {
        "random 100 (default)": sample_background(X_train),
        "random 20": sample_background(X_train, size=20, seed=SEED + 1),
        "k-means 10": shap.kmeans(X_train, 10),
        "benign only (100)": sample_background(benign, size=BACKGROUND_SIZE),
        "malignant only (100)": sample_background(malignant, size=BACKGROUND_SIZE),
    }


def background_sweep(fn, X_train, y_train, X_test, instances):
    """Same instance, same model, different reference distribution -> different phi_0 and attributions."""
    rows, phis = [], {}
    for bg_name, background in background_sets(X_train, y_train).items():
        kernel_data = background.values if isinstance(background, pd.DataFrame) else background
        kernel = shap.KernelExplainer(fn, kernel_data)
        for case, idx in instances.items():
            phi = explain_kernel_shap(kernel, X_test.loc[idx], KERNEL_SHAP_NSAMPLES, SEED)
            phis[(bg_name, case)] = phi
            rows.append({"background": bg_name, "instance": case, "phi_0": float(kernel.expected_value),
                         "top-5": ", ".join(top_k(phi)), "phi": phi})
    default = "random 100 (default)"
    table = []
    for r in rows:
        ref = phis[(default, r["instance"])]
        table.append(
            {k: v for k, v in r.items() if k != "phi"}
            | {
                "sign flips vs default (all 30)": int((np.sign(r["phi"]) != np.sign(ref)).sum()),
                "sign flips vs default (top-5)": int(
                    (np.sign(r["phi"][top_k(ref)]) != np.sign(ref[top_k(ref)])).sum()
                ),
            }
        )
    return pd.DataFrame(table), phis


def plot_lime_boxplots(lime_runs, instances_to_plot, name):
    fig, axes = plt.subplots(1, len(instances_to_plot), figsize=(6.5 * len(instances_to_plot), 5))
    for ax, case in zip(np.atleast_1d(axes), instances_to_plot):
        frame = pd.DataFrame(lime_runs[case][(1000, 0.75)])
        features = frame.abs().mean().sort_values(ascending=False).index[:8]
        long = frame[features].melt(var_name="feature", value_name="LIME weight")
        sns.boxplot(data=long, y="feature", x="LIME weight", ax=ax, color="#9ecae1")
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(f"{case}: LIME weights over {len(frame)} seeds (num_samples=1000)", fontsize=10)
    fig.tight_layout()
    save_figure(fig, name)


def plot_stability_vs_budget(table, name):
    budget = table[table["kernel_width"].isin(["0.75*sqrt(d)", "-"])]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for (method, case), group in budget.groupby(["method", "instance"]):
        style = "-" if method == "LIME" else "--"
        axes[0].plot(group["num_samples"], group["mean_pairwise_jaccard"], style, marker="o", label=f"{method} | {case}")
        axes[1].plot(group["num_samples"], group["runtime_s"], style, marker="o", label=f"{method} | {case}")
    axes[0].set(xscale="log", xlabel="perturbation samples", ylabel="mean pairwise Jaccard@5 across seeds",
                title="Top-5 stability vs sample budget")
    axes[1].set(xscale="log", yscale="log", xlabel="perturbation samples", ylabel="seconds per explanation",
                title="Runtime vs sample budget")
    axes[1].legend(fontsize=7, loc="center left", bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    save_figure(fig, name)


def plot_background_effect(phis, case, name):
    fig, ax = plt.subplots(figsize=(7, 5))
    frame = pd.DataFrame({bg: phi for (bg, c), phi in phis.items() if c == case})
    features = frame["random 100 (default)"].abs().sort_values(ascending=False).index[:8]
    frame.loc[features[::-1]].plot.barh(ax=ax, width=0.8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(f"Kernel SHAP (RF) | {case}: attributions under different backgrounds", fontsize=10)
    ax.legend(fontsize=7)
    fig.tight_layout()
    save_figure(fig, name)


def main():
    X_train, X_test, y_train, y_test = load_split()
    background = sample_background(X_train)
    rf = fit_random_forest(X_train, y_train)
    fn = malignant_probability(rf, list(X_train.columns))
    instances = select_instances(pd.Series(rf.predict_proba(X_test)[:, 1], index=X_test.index), y_test)

    tree_reference, _ = explain_tree(make_tree_explainers(rf, background)["interventional"], X_test.loc[list(instances.values())])
    kernel = shap.KernelExplainer(fn, background.values)

    rows, lime_runs = [], {}
    for case, idx in instances.items():
        x_row = X_test.loc[idx]
        print(f"... sweeping {case}")
        lime_rows, lime_runs[case] = lime_sweep(fn, X_train, x_row, case)
        kernel_rows, _ = kernel_shap_sweep(kernel, x_row, case, tree_reference.loc[idx])
        rows += lime_rows + kernel_rows
    table = pd.DataFrame(rows)
    save_table(table, "exp2_seed_stability", index=False)

    default = table[
        ((table["method"] == "LIME") & (table["num_samples"] == LIME_NUM_SAMPLES) & (table["kernel_width"] == "0.75*sqrt(d)"))
        | ((table["method"] == "Kernel SHAP") & (table["num_samples"] == KERNEL_SHAP_NSAMPLES))
    ]
    least_stable = default.sort_values(["top1_mode_share", "mean_pairwise_jaccard"]).iloc[0]
    print("\nLeast stable explanation at default settings:")
    print(least_stable.to_string())

    plot_lime_boxplots(lime_runs, ["confident_malignant", "borderline"], "exp2_lime_seed_boxplots")
    plot_stability_vs_budget(table, "exp2_stability_vs_budget")

    bg_table, bg_phis = background_sweep(fn, X_train, y_train, X_test, instances)
    save_table(bg_table, "exp2_background", index=False)
    plot_background_effect(bg_phis, "borderline", "exp2_background_borderline")


if __name__ == "__main__":
    main()
