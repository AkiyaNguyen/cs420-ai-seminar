# CS420 Seminar S04 — LIME and SHAP: Code

Experiments for the seminar topic *LIME and SHAP: Explaining Machine-Learning Predictions*.
Guiding question: does a highly accurate prediction necessarily come with a reliable explanation?

Three experiments compare LIME, Kernel SHAP and TreeSHAP on a Random Forest, with interventional TreeSHAP as the exact reference. The seminar paper built from these results is in `CS420-Report/`.

## Contents

| File | Required Tasks | Purpose | Used in the paper |
|---|---|---|---|
| `src/exp1_local_explanations.py` | 3, 4, 5 | 3-feature Shapley sanity check; 3 instances explained by every method; feature-contribution charts, LIME-versus-SHAP table | Section 4.2: contribution and waterfall figures, LIME-versus-SHAP table |
| `src/exp2_stability.py` | 5 | Sensitivity to seed, sample budget, LIME kernel width and SHAP background; runtime; unstable explanation example | Section 4.3: stability and background tables; Appendix C figure |
| `src/exp3_faithfulness.py` | 6 | Deletion curves against a random baseline; correlated features (drop redundant / add noisy copy) to show importance is not causation | Section 5.1: deletion AUC table; Section 5.2: correlated-features table. Its figures are not used |
| `src/common.py` | — | Data split, models, instance selection, explainer wrappers, comparison metrics, figure/table saving | — |
| `CS420-Report/` | 1–6 | LaTeX sources and compiled `main.pdf` of the seminar paper | — |

The scripts write their figures and tables to `report/figures/` and `report/tables/` (prefixed `exp1_`, `exp2_`, `exp3_`). `report/` and `slides/` are kept locally and are not tracked; figures used in the paper are copied into `CS420-Report/img/`.

## Setup

A virtual environment lives in `.venv/` inside this folder. To recreate it:

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt
```

## Running

Run any script from this folder; each writes its figures and tables to `report/` (created on first run) and prints the tables to stdout. Scripts are independent and deterministic (`SEED = 42`). Approximate runtimes: exp1 ~20 s, exp2 ~4 min, exp3 ~3 min.

```bash
.venv/Scripts/python.exe src/exp1_local_explanations.py
.venv/Scripts/python.exe src/exp2_stability.py
.venv/Scripts/python.exe src/exp3_faithfulness.py
```

## Building the paper

From `CS420-Report/`, run `latexmk -pdf main.tex`, or `pdflatex main`, `bibtex main`, then `pdflatex main` twice.

## Conventions

- **Label:** `1 = malignant`, `0 = benign`. sklearn's original encoding is flipped, so probabilities and attribution signs read as "pushing towards malignant".
- **Seed:** `SEED = 42` for the split, the models, LIME and Kernel SHAP sampling.
- **Explained output:** `P(malignant)` of the Random Forest.
- **Background:** 100 random training rows, shared by Kernel SHAP and TreeSHAP (interventional).
- **Kernel SHAP:** always called with `l1_reg=False`; since shap 0.47 the default keeps only 10 features and zeroes the rest.
- **LIME in regression mode:** read weights from `explanation.local_exp[1]`; label `0` holds the negated weights.

## Data

Wolberg, W., Street, W. N. & Mangasarian, O. (1995). *Breast Cancer Wisconsin (Diagnostic)*. UCI Machine Learning Repository. Loaded via `sklearn.datasets.load_breast_cancer` (569 samples, 30 features, 212 malignant / 357 benign).

## References

- Ribeiro, M. T., Singh, S. & Guestrin, C. (2016). "Why Should I Trust You?": Explaining the Predictions of Any Classifier. *KDD 2016*.
- Lundberg, S. M. & Lee, S.-I. (2017). A Unified Approach to Interpreting Model Predictions. *NeurIPS 2017*.
- Mersha, M., Lam, K., Wood, J., AlShami, A. K., & Kalita, J. (2024). Explainable artificial intelligence: A survey of needs, techniques, applications, and future direction. *Neurocomputing*, 599, 128111. [doi:10.1016/j.neucom.2024.128111](https://doi.org/10.1016/j.neucom.2024.128111)
