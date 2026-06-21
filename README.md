<div align="center">

# 🩺 Type 2 Diabetes Prediction & Feature‑Importance Consistency

**An explainable machine‑learning study using NHANES data to predict diabetes and stress‑test how stable "important feature" conclusions really are.**

![Python](https://img.shields.io/badge/Python-3.x-3776AB?logo=python&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Gradient%20Boosting-2C8EBB)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-8A2BE2)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebooks-F37626?logo=jupyter&logoColor=white)
![Status](https://img.shields.io/badge/Status-Research%20%2F%20Educational-lightgrey)

</div>

---

The project compares **Logistic Regression**, **Decision Tree**, **Random Forest**, and **XGBoost** models trained with either **SMOTE oversampling** or **random undersampling**. Explanations are produced with **SHAP**, **permutation importance**, and **model‑specific importance**, then compared using rank correlation and top‑feature overlap.

## 📑 Contents

- [Project objectives](#-project-objectives)
- [Data](#-data)
- [Methods](#-methods)
- [Tools & packages](#-tools--packages)
- [Repository structure](#-repository-structure)
- [Setup](#-setup)
- [Reproducing the analysis](#-reproducing-the-analysis)
- [Notebook guide](#-notebook-guide)
- [Outputs](#-outputs)
- [Reproducibility notes](#-reproducibility-notes)
- [Responsible use](#-responsible-use)

---

## 🎯 Project objectives

This repository addresses three related questions:

| # | Question |
|---|---|
| 1 | How accurately can type 2 diabetes be predicted from **non‑diagnostic** demographic, socioeconomic, lifestyle, dietary, and health variables? |
| 2 | How **consistent** are the most important features across different machine‑learning algorithms? |
| 3 | How much do **class‑imbalance treatment** and **explainability method** affect feature rankings? |

> [!IMPORTANT]
> Outcome‑defining variables — HbA1c, fasting glucose, reported diagnosis, insulin use, and diabetes medication use — are **excluded from the predictors** to avoid target leakage.

## 📊 Data

Built from publicly available **[NHANES](https://www.cdc.gov/nchs/nhanes/)** cycles:

`2011–2012` · `2013–2014` · `2015–2016` · `2017–March 2020 (pre‑pandemic)`

| Detail | Value |
|---|---|
| Study population | 26,000 adults, aged 20+ |
| Exclusions | Pregnant participants; insufficient diabetes‑outcome info |
| Final model features | 43 (after encoding) |
| Train/test split | 80/20, stratified, `random_state=42` |
| Diabetes prevalence | ≈ 18.9% in both sets |

**Diabetes is defined** as meeting **at least one** of:

- ✅ Self‑reported diabetes diagnosis
- ✅ HbA1c ≥ 6.5%
- ✅ Fasting plasma glucose ≥ 126 mg/dL
- ✅ Current insulin use
- ✅ Current diabetes medication use

## 🧪 Methods

### Models

| Model | Input data |
|---|---|
| Logistic Regression | Scaled features |
| Decision Tree | Unscaled features |
| Random Forest | Unscaled features |
| XGBoost | Unscaled features |

Each model is trained **twice**:

- **Strategy A — SMOTE:** synthetic minority‑class oversampling within each training fold
- **Strategy B — Undersampling:** random majority‑class undersampling within each training fold

Resampling happens *inside* the cross‑validation pipeline to prevent leakage. Hyperparameters are chosen via randomized search with 10‑fold stratified CV, optimizing ROC AUC.

### Feature‑importance methods

| Method | Type |
|---|---|
| SHAP | Game‑theoretic attribution |
| Permutation importance | Model‑agnostic |
| Model‑specific importance | Coefficients / impurity reduction / XGBoost gain |

**Consistency** across methods, models, and sampling strategies is measured with:

- Spearman rank correlation
- Top‑10 feature overlap
- Jaccard similarity

## 🧰 Tools & packages

Everything runs in Python via Jupyter notebooks. `pandas` and `NumPy` handle data merging and harmonization, `scikit-learn` drives the modeling pipelines — preprocessing, cross‑validation, hyperparameter search, and evaluation — and SMOTE/undersampling are applied *inside* those pipelines via `imbalanced-learn`. `XGBoost` trains the gradient‑boosted models, `SHAP` produces the explanations, and `Matplotlib`/`Seaborn` handle the plots. A fixed seed of `42` is used wherever randomness is involved (splits, resampling, cross‑validation, training) to keep results reproducible.

| Tool / package | Version |
|---|---|
| Python | 3.13.9 |
| Jupyter Notebook | 7.4.5 |
| pandas | 2.3.3 |
| NumPy | 2.3.5 |
| scikit-learn | 1.7.2 |
| imbalanced-learn | 0.14.0 |
| XGBoost | 3.2.0 |
| SHAP | 0.52.0 |
| Matplotlib | 3.10.6 |
| Seaborn | 0.13.2 |

> [!NOTE]
> Saved model files can be sensitive to package versions — if you hit loading errors after upgrading, match the versions above or retrain from scratch (see [Reproducibility notes](#-reproducibility-notes)).

## 🗂 Repository structure

```text
.
├── data/
│   ├── raw/                 # Merged NHANES data
│   ├── processed/           # Analysis-ready participant dataset
│   └── splits/              # Encoded train/test sets and preprocessing report
├── models/                  # Saved fitted pipelines
├── notebooks/
│   ├── models/               # Eight model-training notebooks
│   ├── 01_data_merging.ipynb
│   ├── 02_data_preparation.ipynb
│   ├── 03_eda.ipynb
│   ├── 04_preprocessing.ipynb
│   ├── 05_model_comparison.ipynb
│   ├── 06_model_feature_consistency.ipynb
│   ├── 07_preprocessing_feature_consistency.ipynb
│   └── 08_xai_method_consistency.ipynb
└── results/
    ├── decision_tree/
    ├── logistic_regression/
    ├── random_forest/
    ├── xgboost/
    └── figures/
```

## ⚙️ Setup

**1. Clone the repository**

```bash
git clone https://github.com/KaylanDG/G2_Feature_Importance_DT2_Prediction.git
cd G2_Feature_Importance_DT2_Prediction
```

**2. Create a virtual environment**

```bash
python -m venv .venv
source .venv/bin/activate
```

On Windows:

```powershell
.venv\Scripts\activate
```

**3. Install dependencies**

```bash
python -m pip install --upgrade pip
python -m pip install \
  jupyter==7.4.5 \
  pandas==2.3.3 \
  numpy==2.3.5 \
  scipy pyarrow \
  matplotlib==3.10.6 \
  seaborn==0.13.2 \
  scikit-learn==1.7.2 \
  imbalanced-learn==0.14.0 \
  xgboost==3.2.0 \
  shap==0.52.0 \
  joblib
```

> [!NOTE]
> Version pins match the [Tools & packages](#-tools--packages) table above. Drop the pins if you'd rather use the latest releases — just be aware saved `.pkl` files may not load across major version changes.

**4. Launch Jupyter**

```bash
jupyter notebook
```

## 🔁 Reproducing the analysis

The repository already contains generated datasets, fitted models, metrics, and figures — the notebooks can be opened and inspected **without retraining anything**.

For a full rebuild, run the notebooks **in this order**:

1. `notebooks/01_data_merging.ipynb`
2. `notebooks/02_data_preparation.ipynb`
3. `notebooks/03_eda.ipynb` — *optional for model reproduction*
4. `notebooks/04_preprocessing.ipynb`
5. All eight notebooks in `notebooks/models/`
6. `notebooks/05_model_comparison.ipynb`
7. `notebooks/06_model_feature_consistency.ipynb`
8. `notebooks/07_preprocessing_feature_consistency.ipynb`
9. `notebooks/08_xai_method_consistency.ipynb`

> [!TIP]
> Run the numbered notebooks from the `notebooks/` directory and the model notebooks from `notebooks/models/` — their file paths are relative to those locations.
>
> The model notebooks reuse fitted `.pkl` files in `models/` when available. Delete the relevant saved model first if you want to repeat hyperparameter tuning from scratch.

## 📓 Notebook guide

| Notebook | Purpose |
|---|---|
| `01_data_merging.ipynb` | Downloads and merges selected NHANES modules across survey cycles |
| `02_data_preparation.ipynb` | Harmonizes variables, constructs the target, applies population criteria, selects predictors |
| `03_eda.ipynb` | Explores class balance, missingness, distributions, and predictor relationships |
| `04_preprocessing.ipynb` | Splits, imputes, encodes, scales, and exports model-ready datasets |
| `models/*_A.ipynb` | Trains each algorithm with SMOTE |
| `models/*_B.ipynb` | Trains each algorithm with random undersampling |
| `05_model_comparison.ipynb` | Compares predictive performance and confusion matrices |
| `06_model_feature_consistency.ipynb` | Compares SHAP rankings across algorithms |
| `07_preprocessing_feature_consistency.ipynb` | Compares SHAP rankings across sampling strategies |
| `08_xai_method_consistency.ipynb` | Compares SHAP, permutation, and model-specific rankings |

## 📤 Outputs

Each model run exports:

- 📦 a fitted pipeline → `models/`
- 📈 test‑set metrics
- 🔢 predictions and probabilities
- 🏆 full feature‑importance rankings
- 🧩 a serialized SHAP explanation

The comparison notebooks generate publication‑ready figures and summary tables under `results/figures/`, including:

- Model performance and confusion‑matrix comparisons
- SHAP beeswarm plots
- Pairwise feature‑ranking heatmaps
- Top‑10 overlap and Jaccard charts
- Preprocessing rank‑movement plots
- XAI‑method consistency summaries

## 🔬 Reproducibility notes

- The train/test split uses `random_state=42`.
- Imputation, encoding, scaling, and resampling are fitted using **training data only**.
- The held‑out test set is **never** resampled.
- Saved model files can be version‑sensitive — if loading fails after a major package upgrade, delete the affected `.pkl` and retrain it.
- NHANES is a complex survey. This project treats the pooled data as a predictive‑modelling dataset and does **not** apply survey weights for population‑level inference.

## ⚠️ Responsible use

> [!CAUTION]
> This project is intended for **research and educational purposes only**. It is **not** a medical diagnostic system and should **not** be used to make clinical decisions. Model associations and feature‑importance scores do **not** establish causality.
