"""
Codveda Technology Internship
Level 2 - Task 2: Classification with Logistic Regression
Author: Peter Molepo
Description: Classify customer churn using Logistic Regression and compare with other classifiers.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report, confusion_matrix, roc_curve, auc
)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titleweight": "bold",
})


# ─────────────────────────────────────────────
#  1. GENERATE CHURN DATASET
# ─────────────────────────────────────────────

def create_dataset():
    """Synthetic customer churn dataset."""
    np.random.seed(42)
    n = 600

    tenure = np.random.randint(1, 72, n)
    monthly_charge = np.random.normal(65, 25, n).clip(20, 120)
    total_charges = tenure * monthly_charge + np.random.normal(0, 50, n)
    contract = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n, p=[0.55, 0.25, 0.20])
    tech_support = np.random.choice([0, 1], n, p=[0.45, 0.55])
    online_security = np.random.choice([0, 1], n, p=[0.50, 0.50])
    num_services = np.random.randint(1, 8, n)

    # Churn probability (logistic-like)
    contract_enc = np.where(contract == "Month-to-Month", 1, 0)
    log_odds = (
        -2.5
        + 0.04 * monthly_charge
        - 0.03 * tenure
        + 1.2 * contract_enc
        - 0.5 * tech_support
        - 0.3 * online_security
        - 0.1 * num_services
    )
    churn_prob = 1 / (1 + np.exp(-log_odds))
    churn = (np.random.rand(n) < churn_prob).astype(int)

    df = pd.DataFrame({
        "Tenure": tenure,
        "MonthlyCharges": monthly_charge.round(2),
        "TotalCharges": total_charges.round(2),
        "Contract": contract,
        "TechSupport": tech_support,
        "OnlineSecurity": online_security,
        "NumServices": num_services,
        "Churn": churn,
    })

    df.to_csv("customer_churn.csv", index=False)
    print(f"\n  Churn rate: {churn.mean() * 100:.1f}%  ({churn.sum()} churned / {n} total)")
    return df


# ─────────────────────────────────────────────
#  2. PREPROCESSING
# ─────────────────────────────────────────────

def preprocess(df):
    print("\n" + "=" * 60)
    print("  STEP 2: PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # One-hot encode Contract
    df = pd.get_dummies(df, columns=["Contract"], drop_first=False)
    bool_cols = df.select_dtypes(bool).columns
    df[bool_cols] = df[bool_cols].astype(int)

    X = df.drop(columns=["Churn"])
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_test_sc = scaler.transform(X_test)

    print(f"  Features after encoding : {list(X.columns)}")
    print(f"  Train samples           : {len(X_train)}")
    print(f"  Test samples            : {len(X_test)}")

    return X_train, X_test, y_train, y_test, X_train_sc, X_test_sc, X.columns


# ─────────────────────────────────────────────
#  3. TRAIN & EVALUATE
# ─────────────────────────────────────────────

def eval_classifier(name, model, X_train, X_test, y_train, y_test, needs_proba=True):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "Predictions": y_pred,
        "Model_Obj": model,
    }

    if needs_proba and hasattr(model, "predict_proba"):
        metrics["Probabilities"] = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, "decision_function"):
        metrics["Probabilities"] = model.decision_function(X_test)

    return metrics


def train_all(X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc):
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL TRAINING")
    print("=" * 60)

    results = []

    results.append(eval_classifier(
        "Logistic Regression",
        LogisticRegression(max_iter=1000, C=1.0, random_state=42),
        X_tr_sc, X_te_sc, y_train, y_test
    ))

    results.append(eval_classifier(
        "Random Forest",
        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        X_train, X_test, y_train, y_test
    ))

    results.append(eval_classifier(
        "Gradient Boosting",
        GradientBoostingClassifier(n_estimators=100, random_state=42),
        X_train, X_test, y_train, y_test
    ))

    results.append(eval_classifier(
        "SVM",
        SVC(kernel="rbf", probability=True, random_state=42),
        X_tr_sc, X_te_sc, y_train, y_test
    ))

    print(f"\n  {'Model':<22} {'Accuracy':>10} {'Precision':>10} {'Recall':>8} {'F1':>8}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['Model']:<22} {r['Accuracy']:>10.4f} {r['Precision']:>10.4f} "
              f"{r['Recall']:>8.4f} {r['F1']:>8.4f}")

    return results


# ─────────────────────────────────────────────
#  4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_results(results, y_test):
    best = max(results, key=lambda x: x["F1"])

    # ── Plot 1: Metrics Comparison ────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Classifier Performance Comparison", fontsize=14, fontweight="bold")

    names = [r["Model"] for r in results]
    metrics = ["Accuracy", "Precision", "Recall", "F1"]
    x = np.arange(len(names))
    width = 0.2
    colors = ["#3498DB", "#E74C3C", "#27AE60", "#F39C12"]

    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [r[metric] for r in results]
        axes[0].bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)

    axes[0].set_xticks(x + width * 1.5)
    axes[0].set_xticklabels(names, rotation=15, ha="right")
    axes[0].set_ylabel("Score")
    axes[0].set_ylim(0, 1.1)
    axes[0].set_title("All Metrics")
    axes[0].legend()

    # F1 only highlight
    f1_vals = [r["F1"] for r in results]
    bar_colors = ["#E74C3C" if n == best["Model"] else "#AED6F1" for n in names]
    axes[1].bar(names, f1_vals, color=bar_colors)
    axes[1].set_title("F1 Score (best in red)")
    axes[1].set_ylabel("F1 Score")
    axes[1].set_ylim(0, 1)
    axes[1].tick_params(axis="x", rotation=15)
    for i, v in enumerate(f1_vals):
        axes[1].text(i, v + 0.01, f"{v:.3f}", ha="center")

    plt.tight_layout()
    plt.savefig("plot_01_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_01_model_comparison.png")

    # ── Plot 2: Confusion Matrix (best model) ─────────────────
    cm = confusion_matrix(y_test, best["Predictions"])
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    plt.colorbar(im, ax=ax)
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["No Churn", "Churn"])
    ax.set_yticklabels(["No Churn", "Churn"])
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    fontsize=16, color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix – {best['Model']}")
    plt.tight_layout()
    plt.savefig("plot_02_confusion_matrix.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_02_confusion_matrix.png")

    # ── Plot 3: ROC Curves ────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    colors_roc = ["#E74C3C", "#3498DB", "#27AE60", "#F39C12"]
    for r, color in zip(results, colors_roc):
        if "Probabilities" in r:
            fpr, tpr, _ = roc_curve(y_test, r["Probabilities"])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, color=color, linewidth=2,
                    label=f"{r['Model']} (AUC = {roc_auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", linewidth=1.5, label="Random Classifier")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves – All Models")
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig("plot_03_roc_curves.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_03_roc_curves.png")

    print(f"\n  Best model: {best['Model']}  (F1 = {best['F1']:.4f})")

    print(f"\n  Detailed Classification Report – {best['Model']}:")
    print(classification_report(y_test, best["Predictions"],
                                 target_names=["No Churn", "Churn"]))


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CODVEDA INTERNSHIP | LEVEL 2 – TASK 2")
    print("  Classification – Customer Churn Prediction")
    print("=" * 60)

    df = create_dataset()
    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, feat_names = preprocess(df)
    results = train_all(X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc)

    print("\n  Generating visualisations...")
    plot_results(results, y_test)

    print("\n  Classification pipeline complete.")


if __name__ == "__main__":
    main()
