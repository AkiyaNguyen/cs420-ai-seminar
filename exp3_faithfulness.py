"""Experiment 3 - Faithfulness and the causality trap.

Part A: deletion curves. Remove features in the order an explanation ranks them (by replacing them with
        background values) and track the predicted-class probability. A faithful ranking makes it fall fast.
        Baseline: random feature order.
Part B: correlated features. Drop the redundant size features, or add a noisy copy of `worst radius`,
        retrain, and watch accuracy stay put while attributions move.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.metrics import accuracy_score, roc_auc_score

from common import (
    KERNEL_SHAP_NSAMPLES,
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
)

EVAL_SIZE = 40
RANDOM_ORDERS = 10
DELETION_NSAMPLES = 1000
DROPPED_FEATURES = ["mean perimeter", "mean area", "worst perimeter", "worst area"]
DUPLICATED_FEATURE = "worst radius"
COPY_NAME = "worst radius (noisy copy)"
SIZE_FEATURES = ["mean radius", "mean perimeter", "mean area", "worst radius", "worst perimeter", "worst area", COPY_NAME]


# ---------------------------------------------------------------------------
# Part A: deletion curves
# ---------------------------------------------------------------------------

def deletion_curve(predict_proba_1, x_row: pd.Series, order: list[str], background: pd.DataFrame, target: int) -> np.ndarray:
    """P(target class) after replacing the first k features of `order` with background values, k = 0..d."""
    batches = []
    for k in range(len(order) + 1):
        rows = background.copy()
        kept = [f for f in x_row.index if f not in order[:k]]
        if kept:
            rows[kept] = np.tile(x_row[kept].values, (len(rows), 1))
        batches.append(rows)
    proba = predict_proba_1(pd.concat(batches, ignore_index=True)).reshape(len(order) + 1, len(background)).mean(axis=1)
    return proba if target == 1 else 1 - proba


def ranking(phi: pd.Series, target: int) -> list[str]:
    """Features ordered by how strongly they push towards the predicted class."""
    return list((phi if target == 1 else -phi).sort_values(ascending=False).index)


def deletion_experiment(rf, fn, X_train, X_test, y_test, background):
    eval_rows = X_test.groupby(y_test, group_keys=False).apply(
        lambda g: g.sample(n=EVAL_SIZE // 2, random_state=SEED)
    )
    deletion_background = background.sample(n=50, random_state=SEED)
    rng = np.random.default_rng(SEED)
    kernel = shap.KernelExplainer(fn, background.values)
    lime_explainer = make_lime_explainer(X_train, seed=SEED)
    tree_phi, _ = explain_tree(make_tree_explainers(rf, background)["interventional"], eval_rows)
    predict_1 = lambda X: rf.predict_proba(X)[:, 1]

    curves = []
    for idx, x_row in eval_rows.iterrows():
        target = int(rf.predict(x_row.to_frame().T)[0])
        phis = {
            "LIME": explain_lime(lime_explainer, fn, x_row)["phi"],
            "Kernel SHAP": explain_kernel_shap(kernel, x_row, DELETION_NSAMPLES, SEED),
            "TreeSHAP (interventional)": tree_phi.loc[idx],
        }
        for method, phi in phis.items():
            curves.append({"method": method, "index": idx,
                           "curve": deletion_curve(predict_1, x_row, ranking(phi, target), deletion_background, target)})
        random_curves = [
            deletion_curve(predict_1, x_row, list(rng.permutation(x_row.index)), deletion_background, target)
            for _ in range(RANDOM_ORDERS)
        ]
        curves.append({"method": "Random order (baseline)", "index": idx, "curve": np.mean(random_curves, axis=0)})
    return curves


def summarize_deletion(curves) -> pd.DataFrame:
    rows = []
    for method, group in pd.DataFrame(curves).groupby("method"):
        stacked = np.vstack(group["curve"])
        rows.append(
            {
                "method": method,
                "instances": len(stacked),
                "P(pred) at k=0": stacked[:, 0].mean(),
                "P(pred) after top-1 removed": stacked[:, 1].mean(),
                "P(pred) after top-3 removed": stacked[:, 3].mean(),
                "P(pred) after top-5 removed": stacked[:, 5].mean(),
                "deletion AUC (lower = more faithful)": np.trapezoid(stacked, dx=1 / (stacked.shape[1] - 1), axis=1).mean(),
            }
        )
    return pd.DataFrame(rows).sort_values("deletion AUC (lower = more faithful)")


def plot_deletion(curves, name):
    frame = pd.DataFrame(curves)
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for method, group in frame.groupby("method"):
        mean_curve = np.vstack(group["curve"]).mean(axis=0)
        style = ":" if method.startswith("Random") else "-"
        ax.plot(range(len(mean_curve)), mean_curve, style, marker=".", label=method)
    ax.set(title=f"RF: deletion curve (mean over {frame['index'].nunique()} test instances)",
           xlabel="number of top-ranked features replaced by background", ylabel="P(predicted class)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    save_figure(fig, name)


# ---------------------------------------------------------------------------
# Part B: correlated features
# ---------------------------------------------------------------------------

def add_noisy_copy(X: pd.DataFrame, reference_std: float, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = X.copy()
    X[COPY_NAME] = X[DUPLICATED_FEATURE] + rng.normal(0, 0.05 * reference_std, len(X))
    return X


def importance_profile(X_train, X_test, y_train, y_test, instance_idx) -> tuple[dict, pd.DataFrame]:
    """RF accuracy plus global (mean|phi| on test) and local (one instance) attributions."""
    background = sample_background(X_train)
    rf = fit_random_forest(X_train, y_train)
    fn = malignant_probability(rf, list(X_train.columns))
    x_row = X_test.loc[instance_idx]

    tree = make_tree_explainers(rf, background)["interventional"]
    kernel = shap.KernelExplainer(fn, background.values)
    lime_explainer = make_lime_explainer(X_train, seed=SEED)

    profile = pd.DataFrame(
        {
            "global mean|phi| (TreeSHAP)": explain_tree(tree, X_test)[0].abs().mean(),
            "local TreeSHAP": explain_tree(tree, x_row.to_frame().T)[0].iloc[0],
            "local Kernel SHAP": explain_kernel_shap(kernel, x_row, KERNEL_SHAP_NSAMPLES, SEED),
            "local LIME": explain_lime(lime_explainer, fn, x_row)["phi"],
        }
    )
    metrics = {
        "accuracy": accuracy_score(y_test, rf.predict(X_test)),
        "ROC AUC": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
    }
    return metrics, profile


def correlated_features_experiment(X_train, X_test, y_train, y_test, instance_idx):
    reference_std = X_train[DUPLICATED_FEATURE].std()
    variants = {
        "full (30 features)": (X_train, X_test),
        "without perimeter/area": (X_train.drop(columns=DROPPED_FEATURES), X_test.drop(columns=DROPPED_FEATURES)),
        "with noisy copy of worst radius": (
            add_noisy_copy(X_train, reference_std, SEED), add_noisy_copy(X_test, reference_std, SEED + 1)
        ),
    }
    metric_rows, profiles = {}, {}
    for name, (Xtr, Xte) in variants.items():
        metric_rows[name], profiles[name] = importance_profile(Xtr, Xte, y_train, y_test, instance_idx)

    print(f"\ncorrelation of {DUPLICATED_FEATURE} with its noisy copy: "
          f"{variants['with noisy copy of worst radius'][0][[DUPLICATED_FEATURE, COPY_NAME]].corr().iloc[0, 1]:.4f}")
    save_table(pd.DataFrame(metric_rows).T, "exp3_correlated_metrics")

    long = []
    for variant, profile in profiles.items():
        for feature in SIZE_FEATURES:
            row = {"variant": variant, "feature": feature}
            row |= {col: (profile.loc[feature, col] if feature in profile.index else np.nan) for col in profile.columns}
            long.append(row)
        row = {"variant": variant, "feature": "SUM over size features"}
        row |= {col: profile.loc[profile.index.intersection(SIZE_FEATURES), col].abs().sum() for col in profile.columns}
        long.append(row)
    table = pd.DataFrame(long)
    save_table(table, "exp3_correlated_attributions", index=False)
    return table


def plot_correlated(table, name):
    columns = ["global mean|phi| (TreeSHAP)", "local TreeSHAP", "local Kernel SHAP", "local LIME"]
    features = [f for f in SIZE_FEATURES]
    fig, axes = plt.subplots(1, len(columns), figsize=(5 * len(columns), 4.5), sharey=True)
    for ax, column in zip(axes, columns):
        pivot = table[table["feature"].isin(features)].pivot(index="feature", columns="variant", values=column)
        pivot.loc[features[::-1]].plot.barh(ax=ax, width=0.8, legend=ax is axes[0])
        ax.axvline(0, color="black", linewidth=0.8)
        ax.set_title(column, fontsize=10)
    axes[0].legend(fontsize=7, loc="lower right")
    fig.suptitle("Size features: attributions move when redundant features are removed or duplicated", y=1.02)
    fig.tight_layout()
    save_figure(fig, name)


def main():
    X_train, X_test, y_train, y_test = load_split()
    background = sample_background(X_train)
    rf = fit_random_forest(X_train, y_train)
    fn = malignant_probability(rf, list(X_train.columns))

    curves = deletion_experiment(rf, fn, X_train, X_test, y_test, background)
    save_table(summarize_deletion(curves), "exp3_deletion_summary", index=False)
    plot_deletion(curves, "exp3_deletion_curves")

    instances = select_instances(pd.Series(rf.predict_proba(X_test)[:, 1], index=X_test.index), y_test)
    table = correlated_features_experiment(X_train, X_test, y_train, y_test, instances["confident_malignant"])
    plot_correlated(table, "exp3_correlated_features")


if __name__ == "__main__":
    main()
