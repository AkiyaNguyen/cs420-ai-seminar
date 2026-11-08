"""Shared configuration and helpers for the seminar experiment scripts."""

import time
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from lime.lime_tabular import LimeTabularExplainer
from scipy.stats import spearmanr
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

SEED = 42
CLASS_NAMES = ["benign", "malignant"]
TOP_K = 5
BACKGROUND_SIZE = 100
LIME_NUM_SAMPLES = 5000
KERNEL_SHAP_NSAMPLES = 2000

CODE_DIR = Path(__file__).resolve().parent.parent
REPORT_DIR = CODE_DIR / "report"
FIGURE_DIR = REPORT_DIR / "figures"
TABLE_DIR = REPORT_DIR / "tables"

sns.set_theme(style="whitegrid")
pd.set_option("display.precision", 4)
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 30)


# ---------------------------------------------------------------------------
# Data and models
# ---------------------------------------------------------------------------

def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Breast Cancer Wisconsin with labels flipped so that 1 = malignant."""
    raw = load_breast_cancer(as_frame=True)
    return raw.data, (1 - raw.target).rename("malignant")


def load_split():
    X, y = load_dataset()
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)


def sample_background(X_train: pd.DataFrame, size: int = BACKGROUND_SIZE, seed: int = SEED) -> pd.DataFrame:
    return X_train.sample(n=min(size, len(X_train)), random_state=seed)


def fit_random_forest(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    return RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1).fit(X, y)


def array_fn(predict, columns):
    """Wrap a fitted sklearn method so explainers can pass plain arrays without feature-name warnings."""

    def wrapped(values):
        return predict(pd.DataFrame(np.atleast_2d(values), columns=columns))

    return wrapped


def malignant_probability(model, columns):
    """The explained output, P(malignant), as a function of a plain array."""
    return array_fn(lambda X: model.predict_proba(X)[:, 1], columns)


def select_instances(proba: pd.Series, y_true: pd.Series) -> dict[str, object]:
    """Test instances explained throughout the experiments, chosen from the RF probabilities.

    - confident_benign / confident_malignant: correct and most confident for each class.
    - borderline: closest to the 0.5 threshold.
    """
    pred = (proba >= 0.5).astype(int)
    correct = pred == y_true
    instances = {
        "confident_benign": proba[correct & (y_true == 0)].idxmin(),
        "confident_malignant": proba[correct & (y_true == 1)].idxmax(),
    }
    distance = (proba - 0.5).abs().drop(list(instances.values()))
    instances["borderline"] = distance.idxmin()
    return instances


# ---------------------------------------------------------------------------
# Explainers. Every function returns attributions as a pd.Series indexed by feature name.
# ---------------------------------------------------------------------------

def make_lime_explainer(X_train: pd.DataFrame, seed: int, kernel_width: float | None = None) -> LimeTabularExplainer:
    """LIME with library defaults (quartile bins, kernel width 0.75 * sqrt(d)) in regression mode."""
    return LimeTabularExplainer(
        X_train.values,
        feature_names=list(X_train.columns),
        mode="regression",
        discretize_continuous=True,
        discretizer="quartile",
        kernel_width=kernel_width,
        random_state=seed,
    )


def explain_lime(explainer: LimeTabularExplainer, fn, x_row: pd.Series, num_samples: int = LIME_NUM_SAMPLES) -> dict:
    """LIME weights for all features, plus the surrogate's prediction g(x) and weighted R^2."""
    columns = list(x_row.index)
    explanation = explainer.explain_instance(
        x_row.values, fn, num_features=len(columns), num_samples=num_samples
    )
    # In regression mode LIME stores the explanation under label 1; label 0 holds the negated weights.
    weights = pd.Series({columns[i]: w for i, w in explanation.local_exp[1]})[columns]
    return {
        "phi": weights,
        "surrogate_prediction": float(np.ravel(explanation.local_pred)[0]),
        "surrogate_r2": float(explanation.score),
        "intercept": float(np.ravel([explanation.intercept[1]])[0]),
    }


def explain_kernel_shap(explainer: shap.KernelExplainer, x_row: pd.Series, nsamples: int, seed: int) -> pd.Series:
    """Kernel SHAP with all features kept (the library default since shap 0.47 keeps only 10)."""
    np.random.seed(seed)
    values = explainer.shap_values(x_row.values.reshape(1, -1), nsamples=nsamples, l1_reg=False, silent=True)
    return pd.Series(np.ravel(values), index=x_row.index)


def positive_class(values) -> np.ndarray:
    """Select the malignant-class slice of TreeExplainer output, whichever layout shap returns."""
    if isinstance(values, list):
        return np.asarray(values[1])
    values = np.asarray(values)
    return values[..., 1] if values.ndim == 3 or (values.ndim == 1 and values.size == 2) else values


def make_tree_explainers(model, background: pd.DataFrame) -> dict[str, shap.TreeExplainer]:
    """TreeSHAP in its two flavours: interventional (same background as Kernel SHAP) and path-dependent."""
    return {
        "interventional": shap.TreeExplainer(
            model, data=background, feature_perturbation="interventional", model_output="probability"
        ),
        "path_dependent": shap.TreeExplainer(model, feature_perturbation="tree_path_dependent"),
    }


def explain_tree(explainer: shap.TreeExplainer, X_rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    values = positive_class(explainer.shap_values(X_rows, check_additivity=False))
    base = float(positive_class(np.atleast_1d(explainer.expected_value)).ravel()[-1])
    return pd.DataFrame(np.atleast_2d(values), index=X_rows.index, columns=X_rows.columns), base


def timed(fn, *args, **kwargs):
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    return result, time.perf_counter() - start


# ---------------------------------------------------------------------------
# Comparison metrics
# ---------------------------------------------------------------------------

def top_k(phi: pd.Series, k: int = TOP_K) -> list[str]:
    return list(phi.abs().sort_values(ascending=False).index[:k])


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    return len(a & b) / len(a | b)


def rank_correlation(phi_a: pd.Series, phi_b: pd.Series) -> float:
    """Spearman correlation of |attribution| over all features."""
    return float(spearmanr(phi_a.abs(), phi_b.abs()[phi_a.index]).statistic)


def sign_agreement(phi: pd.Series, reference: pd.Series, k: int = TOP_K) -> float:
    """Share of the reference's top-k features whose attribution sign matches the reference."""
    features = top_k(reference, k)
    return float((np.sign(phi[features]) == np.sign(reference[features])).mean())


def stability_summary(runs: list[pd.Series], k: int = TOP_K) -> dict:
    """How much repeated explanations of the same instance agree with each other."""
    frame = pd.DataFrame(runs)
    tops = [top_k(phi, k) for phi in runs]
    top1 = pd.Series([t[0] for t in tops])
    union = sorted(set().union(*tops))
    signs = np.sign(frame[union])
    return {
        "mean_pairwise_jaccard": float(np.mean([jaccard(a, b) for a, b in combinations(tops, 2)])),
        "top1_mode": top1.mode()[0],
        "top1_mode_share": float(top1.value_counts(normalize=True).iloc[0]),
        "distinct_top1": int(top1.nunique()),
        "sign_flip_features": int((signs.nunique() > 1).sum()),
        "mean_std": float(frame.std().mean()),
        "mean_abs_phi": float(frame.abs().mean().mean()),
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def save_figure(fig: plt.Figure, name: str) -> Path:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURE_DIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[figure] {path.relative_to(CODE_DIR).as_posix()}")
    return path


def save_table(df: pd.DataFrame, name: str, index: bool = True) -> Path:
    """Save a table as CSV and echo it to stdout so the run log doubles as a summary."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLE_DIR / f"{name}.csv"
    df.to_csv(path, index=index, float_format="%.6g")
    print(f"\n[table] {path.relative_to(CODE_DIR).as_posix()}")
    print(df.to_string(index=index))
    return path


def plot_attribution_bars(ax, phi: pd.Series, title: str, k: int = 10) -> None:
    """Horizontal bars of the top-k |attribution| features; red pushes towards malignant, blue towards benign."""
    top = phi[top_k(phi, k)][::-1]
    colors = ["#d62728" if v > 0 else "#1f77b4" for v in top]
    ax.barh(top.index, top.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=8)
