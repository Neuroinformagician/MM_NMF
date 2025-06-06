# MM_NMF

This repository demonstrates how to analyze myasthenia gravis data with Non-negative Matrix Factorization (NMF) and evaluate predictive models.

## Repository structure

- `step1_load_data_and_NMF.py` – Load the dataset, scale features, perform NMF and save the decomposed matrices.
- `step2_prediction_with_NMF.py` – Use the NMF results to train and evaluate classifiers.
- `All_step.ipynb` – Notebook that combines the steps for interactive execution.
- `mm_utils.py` – Utility functions for loading data and clustering.

## All_step workflow

1. **Prepare data**
   - Place the CSV files in the `data/` directory (`df_3rd.csv`, `df_4th.csv`).
2. **Run Step 1**: `python step1_load_data_and_NMF.py`
   - Loads the dataset and selects clinical score columns.
   - Scales values and performs NMF with four components.
   - Saves the NMF model and matrices under `out/`.
3. **Run Step 2**: `python step2_prediction_with_NMF.py`
   - Loads the saved matrices.
   - Splits the data with cross-validation and trains SVM, logistic regression, random forest and Naive Bayes models.
   - Saves ROC curve data, optimal cutoffs and trained models to `out/`.
4. **Inspect results**
   - Use the CSV and pickle files in `out/` or open `All_step.ipynb` to see the process in a single notebook.

Running the two scripts in order reproduces the entire "All_step" pipeline.
