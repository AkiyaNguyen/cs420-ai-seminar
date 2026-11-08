"""Experiment 1 - Local explanations: do LIME, Kernel SHAP and TreeSHAP agree on the same prediction?

Part A: Shapley values by hand on a 3-feature Random Forest (all 8 coalitions), checked against
        interventional TreeSHAP and Kernel SHAP.
Part B: Random Forest, three test instances, each explained by every method. Produces the
        feature-contribution chart and the LIME-versus-SHAP comparison table.
"""

from itertools import combinations
from math import factorial

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
    jaccard,
    load_split,
    make_lime_explainer,
    make_tree_explainers,
    malignant_probability,
    plot_attribution_bars,
    rank_correlation,
    sample_background,
    save_figure,
    save_table,
    select_instances,
    sign_agreement,
    timed,
    top_k,
)

WORKED_FEATURES = ["worst radius", "worst texture", "worst concave points"]
REFERENCE = "TreeSHAP (interventional)"


# ---------------------------------------------------------------------------
# Part A: worked example
# ---------------------------------------------------------------------------

def worked_example(X_train, y_train, x_row, background):
    """Exact Shapley values of a 3-feature RF, enumerating v(S) = E_B[f(x_S, X_rest)] on P(malignant)."""
    features = WORKED_FEATURES
    rf = fit_random_forest(X_train[features], y_train)
    x = x_row[features]
    n = len(features)

    def value(coalition):
        rows = background[features].copy()
        for feature in coalition:
            rows[feature] = x[feature]
        return float(rf.predict_proba(rows)[:, 1].mean())

    coalitions = [c for size in range(n + 1) for c in combinations(features, size)]
    v = {c: value(c) for c in coalitions}
    save_table(
        pd.DataFrame({"coalition S": ["{" + ", ".join(c) + "}" for c in coalitions], "v(S)": list(v.values())}),
        "exp1_worked_coalitions",
        index=False,
    )

    by_hand = {}
    for feature in features:
        others = [f for f in features if f != feature]
        total = 0.0
        for size in range(n):
            for subset in combinations(others, size):
                weight = factorial(size) * factorial(n - size - 1) / factorial(n)
                with_feature = tuple(f for f in features if f in subset or f == feature)
                total += weight * (v[with_feature] - v[subset])
        by_hand[feature] = total

    tree_phi, tree_base = explain_tree(make_tree_explainers(rf, background[features])["interventional"], x.to_frame().T)
    kernel = shap.KernelExplainer(malignant_probability(rf, features), background[features].values)
    kernel_phi = explain_kernel_shap(kernel, x, nsamples=100, seed=SEED)

    save_table(
        pd.DataFrame(
            {
                "x_j": x,
                "phi_j by hand": pd.Series(by_hand),
                "phi_j TreeSHAP (interventional)": tree_phi.iloc[0],
                "phi_j Kernel SHAP": kernel_phi,
            }
        ),
        "exp1_worked_shapley",
    )

    f_x = float(rf.predict_proba(x.to_frame().T)[0, 1])
    base = v[()]
    save_table(
        pd.DataFrame(
            {
                "quantity": ["phi_0 = v({})", "TreeSHAP base", "sum phi_j", "phi_0 + sum phi_j", "f(x) = P(malignant)"],
                "value": [base, tree_base, sum(by_hand.values()), base + sum(by_hand.values()), f_x],
            }
        ),
        "exp1_worked_local_accuracy",
        index=False,
    )


# ---------------------------------------------------------------------------
# Part B: local explanations of the RF
# ---------------------------------------------------------------------------

def explain_all(rf, fn, X_train, X_test, instances, background):
    """Every method on every instance. Returns a list of result dicts with attributions."""
    kernel = shap.KernelExplainer(fn, background.values)
    tree = make_tree_explainers(rf, background)
    lime_explainer = make_lime_explainer(X_train, seed=SEED)

    results = []
    for case, idx in instances.items():
        x_row = X_test.loc[idx]
        f_x = float(fn(x_row.values)[0])

        def record(method, phi, base, seconds, reconstruction=None, r2=np.nan):
            reconstructed = base + phi.sum() if reconstruction is None else reconstruction
            results.append(
                {
                    "instance": case, "method": method, "phi": phi, "f(x)": f_x, "base": base,
                    "reconstruction": reconstructed, "additivity_gap": abs(reconstructed - f_x),
                    "surrogate_r2": r2, "runtime_s": seconds,
                }
            )

        for flavour, explainer in tree.items():
            (frame, base), seconds = timed(explain_tree, explainer, x_row.to_frame().T)
            record(f"TreeSHAP ({flavour.replace('_', '-')})", frame.iloc[0], base, seconds)

        phi, seconds = timed(explain_kernel_shap, kernel, x_row, KERNEL_SHAP_NSAMPLES, SEED)
        record("Kernel SHAP", phi, float(kernel.expected_value), seconds)

        lime_result, seconds = timed(explain_lime, lime_explainer, fn, x_row)
        record(
            "LIME", lime_result["phi"], lime_result["intercept"], seconds,
            reconstruction=lime_result["surrogate_prediction"], r2=lime_result["surrogate_r2"],
        )
    return results


def comparison_table(results) -> pd.DataFrame:
    reference = {r["instance"]: r["phi"] for r in results if r["method"] == REFERENCE}
    rows = []
    for r in results:
        ref = reference[r["instance"]]
        rows.append(
            {
                "instance": r["instance"],
                "method": r["method"],
                "top-5 features": ", ".join(top_k(r["phi"])),
                "Jaccard@5 vs ref": jaccard(top_k(r["phi"]), top_k(ref)),
                "Spearman |phi| vs ref": rank_correlation(r["phi"], ref),
                "sign agree@5 vs ref": sign_agreement(r["phi"], ref),
                "f(x)": r["f(x)"],
                "base (phi_0 / LIME intercept)": r["base"],
                "reconstruction": r["reconstruction"],
                "additivity gap": r["additivity_gap"],
                "LIME R^2": r["surrogate_r2"],
                "runtime (s)": r["runtime_s"],
            }
        )
    return pd.DataFrame(rows)


def plot_contributions(results, methods, instances, name):
    lookup = {(r["instance"], r["method"]): r for r in results}
    fig, axes = plt.subplots(len(instances), len(methods), figsize=(5 * len(methods), 3.2 * len(instances)))
    for row, case in enumerate(instances):
        for col, method in enumerate(methods):
            r = lookup[(case, method)]
            plot_attribution_bars(axes[row, col], r["phi"], f"{case} | {method}\nf(x)={r['f(x)']:.3f}")
    fig.suptitle("RF: top-10 attributions of P(malignant) (red = towards malignant)", y=1.0)
    fig.tight_layout()
    save_figure(fig, name)


def plot_waterfall(result, x_row, name):
    explanation = shap.Explanation(
        values=result["phi"].values, base_values=result["base"], data=x_row.values,
        feature_names=list(x_row.index),
    )
    shap.plots.waterfall(explanation, max_display=10, show=False)
    fig = plt.gcf()
    fig.suptitle(f"RF {result['method']} | {result['instance']}", y=1.02)
    save_figure(fig, name)


def main():
    X_train, X_test, y_train, y_test = load_split()
    background = sample_background(X_train)
    rf = fit_random_forest(X_train, y_train)
    fn = malignant_probability(rf, list(X_train.columns))

    save_table(
        pd.DataFrame(
            {
                "RF": {
                    "accuracy": accuracy_score(y_test, rf.predict(X_test)),
                    "ROC AUC": roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1]),
                    "test errors": int((rf.predict(X_test) != y_test).sum()),
                }
            }
        ).T,
        "exp1_model_metrics",
    )

    rf_proba = pd.Series(rf.predict_proba(X_test)[:, 1], index=X_test.index)
    instances = select_instances(rf_proba, y_test)
    save_table(
        pd.DataFrame(
            {
                "test index": list(instances.values()),
                "true label": [y_test[i] for i in instances.values()],
                "RF P(malignant)": [rf_proba[i] for i in instances.values()],
            },
            index=list(instances),
        ),
        "exp1_instances",
    )

    worked_example(X_train, y_train, X_test.loc[instances["borderline"]], background)

    results = explain_all(rf, fn, X_train, X_test, instances, background)
    save_table(comparison_table(results), "exp1_comparison", index=False)
    save_table(
        pd.DataFrame([{"instance": r["instance"], "method": r["method"], **r["phi"]} for r in results]),
        "exp1_attributions",
        index=False,
    )
    plot_contributions(results, [REFERENCE, "Kernel SHAP", "LIME"], instances, "exp1_rf_contributions")
    for r in results:
        if r["method"] == "Kernel SHAP" and r["instance"] == "borderline":
            plot_waterfall(r, X_test.loc[instances["borderline"]], "exp1_rf_waterfall_borderline")


if __name__ == "__main__":
    main()
