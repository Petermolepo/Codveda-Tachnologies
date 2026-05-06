# Level 2 – Intermediate

## Task 1: Predictive Modelling (Regression)

**File:** `task1_regression/regression.py`

**What it does:**
- Generates a synthetic house price dataset (500 rows, SA context)
- Splits into 80/20 train/test
- Trains and compares 5 models: Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting
- Evaluates using RMSE, MAE and R²
- Plots model comparison, actual vs predicted, and feature importance

**Outputs:**
- `house_prices.csv`
- `plot_01_model_comparison.png`
- `plot_02_actual_vs_predicted.png`
- `plot_03_feature_importance.png`

**Run:**
```bash
python regression.py
```

---

## Task 2: Classification with Logistic Regression

**File:** `task2_classification/classification.py`

**What it does:**
- Generates a customer churn dataset (600 rows)
- Preprocesses: one-hot encoding, StandardScaler
- Trains and compares 4 classifiers: Logistic Regression, Random Forest, Gradient Boosting, SVM
- Evaluates using accuracy, precision, recall, F1, confusion matrix, and ROC curves

**Outputs:**
- `customer_churn.csv`
- `plot_01_model_comparison.png`
- `plot_02_confusion_matrix.png`
- `plot_03_roc_curves.png`

**Run:**
```bash
python classification.py
```
