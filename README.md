
# Kernel SVMs for Credit Default Prediction under Data-Deletion Requests

**Individual Project | ES 678**

This project investigates Support Vector Machines (SVMs) for credit default prediction and examines their behaviour under data-deletion requests. It compares different predictive models and evaluates the effect of removing selected training samples from an RBF kernel SVM.

The prediction comparison and final test evaluation are documented in `results/final_test.csv`. The corresponding test predictions and fixed data split are also saved in the `results/` directory.

## Reproducing the Experiments

**Environment:** Python 3.12

Install the required dependencies using:

```powershell
python -m pip install -r requirements.txt
```

Run the following scripts in the specified order from the project root directory:

```powershell
python src/data.py
python src/eda.py
python src/baselines.py
python src/models.py
python src/final_eval.py
python src/deletion.py
```

The experimental setup uses a deterministic, stratified 60/20/20 train-validation-test split with random seed 42.

- `data.py` generates the data split and saves the original spreadsheet row identifiers in `results/split_indices.csv`, excluding the header row.
- Exploratory data analysis (EDA) and model selection are performed using training data only.
- The validation set is used for probability calibration and decision-threshold selection.
- The final test set is reserved for locked evaluation. The original test evaluation was performed on **22 October 2026**.

**Note:** Running `final_eval.py` again recomputes the test evaluation and overwrites its output. Therefore, the committed result files should be treated as the record of the original locked evaluation.

### Model Selection and Training

- **RBF SVM:** Hyperparameter search is performed using a stratified 6,000-row subsample of the training set, with 5-fold cross-validation. The selected hyperparameters are subsequently used to refit the model on all 18,000 training samples.
- **Linear SVM:** Hyperparameter selection uses 5-fold cross-validation on the complete training set.
- **Gradient Boosting:** Model selection uses 5-fold cross-validation on the complete training set.
- **Logistic Regression:** The regularization parameter \(C=1\) is selected based on baseline cross-validation.

The validation set is not used to fit the base estimators.

## Data-Deletion Experiment

The deletion experiment examines the effect of removing two types of training samples from the RBF SVM:

1. A non-support vector.
2. A support vector.

After removing each sample, the SVC is retrained and its decision scores are compared against those of the original model on the fixed test set.

To isolate the SVC's dependence on individual training records, the experiment keeps the fitted preprocessing pipeline and explicit RBF gamma value fixed during retraining.

For the observed non-support-vector removal, the predicted test class labels remained unchanged, although small changes in decision scores were observed. This is an empirical observation rather than a claim of exact score equality.

This experiment represents a controlled deletion analysis, not a complete machine-unlearning or data-erasure workflow. In particular, a complete retraining process that also refits the scaler may produce different results because the original scaling statistics were calculated using the full training set.

The observed prediction changes and retraining times are recorded in `results/deletion_audit.csv`.

## Main Experimental Results

The following table presents the test-set Precision-Recall Area Under the Curve (PR-AUC) for each evaluated model.

| Model | Test PR-AUC |
|---|---:|
| Logistic Regression | 0.542 |
| Linear SVM | 0.544 |
| RBF SVM | 0.532 |
| Gradient Boosting | 0.567 |

Additional findings:

- The bootstrap confidence interval for the RBF SVM minus Logistic Regression PR-AUC difference is **[-0.023, +0.004]**.
- The expected-cost difference at the prespecified threshold of \(1/6\) is **[+0.044, +0.107] cost units per customer**.
- The RBF SVM uses **7,016 support vectors**, representing approximately **39.0%** of the training samples.
- The complete RBF SVM pipeline fitting time was **30.9 seconds** on the recorded machine.

For the complete experimental protocol, confidence intervals, interpretation of results, and references, refer to `paper.tex`.

## Dataset and License

The experiments use the **Default of Credit Card Clients** dataset obtained from the UCI Machine Learning Repository (Dataset ID: 350).

- **Source file:** `data/raw/default of credit card clients.xls`
- **Repository:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/350/default%2Bof%2Bcredit%2Bcard%2Bclients)
- **Original reference:** Yeh and Lien (2009).

Please refer to the UCI dataset page for the original dataset citation and licensing information.

This dataset is a historical benchmark and should not be considered suitable for direct deployment in present-day credit decision-making systems. The experimental results do not establish current real-world predictive performance, fairness, or legal compliance.
