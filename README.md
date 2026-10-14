# Cook County House Price Prediction

Hands-on Machine Learning group project.

**Members:** Yuming ZHAI, Sakura, Kayla

## Objective

Predict sale prices of single-card, single-family properties in
Cook County using property and contextual features.

We compare Ridge Regression, Random Forest and Gradient Boosting
using the same 81 input features and eligible records.

## Data

Download the original dataset:
[training_data.parquet](https://ccao-data-public-us-east-1.s3.amazonaws.com/models/inputs/res/2026/run_id=2026-07-30-ecstatic-carly/training_data.parquet)
[data_source](https://github.com/ccao-data/model-res-avm#getting-data)

Save it as `data/training_data.parquet`.

The original dataset contains 405,195 rows and 203 columns.
Raw and processed datasets are not included in this repository.

## Project Files

- `notebooks/01_EDA.ipynb`: data exploration and modelling decisions.
- `notebooks/02_Preprocessing.ipynb`: shared filtering, splitting and imputation.
- `notebooks/03a_Ridge.ipynb`: Ridge training, tuning and evaluation.
- `notebooks/03c_GradientBoosting.ipynb`: Gradient Boosting experiments.
- `shared_preprocessing.py`: reusable preprocessing and encoding functions.
- `results/`: exported model metrics and analysis summaries.

## How to Run

Required packages: pandas, numpy, scikit-learn, pyarrow and matplotlib.

1. Download the dataset to `data/training_data.parquet`.
2. Open JupyterLab and run `notebooks/02_Preprocessing.ipynb`.
   This generates the shared datasets in `data/stage2_outputs/`.
3. Run the model notebooks from the `notebooks` directory.

The EDA notebook documents the initial exploration.
Its data-loading path may need adjustment for your local setup.

## Evaluation Protocol

| Split | Sale years | Eligible records |
|---|---|---:|
| Training | 2017–2023 | 276,685 |
| Validation | 2024 | 30,545 |
| Test | 2025 | 31,535 |

Learned preprocessing is fitted on training data only.
Parameters are selected using validation data.
The test set is used after model choices are fixed.

Metrics: MAE, RMSE and R² on the original price scale.

Historical availability has not been verified for every contextual
feature; this limitation is documented in the notebooks.

## Final Deliverables

One integrated notebook containing code and the English report,
and one PDF slide deck. Working notebooks will be combined before
submission.
