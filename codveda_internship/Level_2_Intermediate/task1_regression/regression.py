"""
Codveda Technology Internship
Level 2 - Task 1: Predictive Modelling (Regression)
Author: Peter Molepo
Description: Build and evaluate regression models to predict house prices.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titleweight": "bold",
})


# ─────────────────────────────────────────────
#  1. GENERATE HOUSE PRICE DATASET
# ─────────────────────────────────────────────

def create_dataset():
    """Synthetic house price dataset with realistic features."""
    np.random.seed(42)
    n = 500

    size_sqm = np.random.normal(180, 60, n).clip(60, 450)
    bedrooms = np.random.choice([1, 2, 3, 4, 5], n, p=[0.05, 0.20, 0.40, 0.25, 0.10])
    bathrooms = np.clip(bedrooms - 1 + np.random.randint(0, 2, n), 1, 4)
    age_years = np.random.randint(0, 40, n)
    garage = np.random.choice([0, 1, 2], n, p=[0.25, 0.50, 0.25])
    location_score = np.random.uniform(1, 10, n)   # neighbourhood desirability

    # Price formula with some noise
    price = (
        500_000
        + 3_800 * size_sqm
        + 75_000 * bedrooms
        + 50_000 * bathrooms
        - 8_000 * age_years
        + 60_000 * garage
        + 25_000 * location_score
        + np.random.normal(0, 80_000, n)
    ).clip(250_000, 4_500_000)

    df = pd.DataFrame({
        "SizeSqm": size_sqm.round(1),
        "Bedrooms": bedrooms,
        "Bathrooms": bathrooms,
        "AgeYears": age_years,
        "GarageSpaces": garage,
        "LocationScore": location_score.round(2),
        "Price": price.round(-3),
    })

    df.to_csv("house_prices.csv", index=False)
    return df


# ─────────────────────────────────────────────
#  2. DATA PREPARATION
# ─────────────────────────────────────────────

def prepare_data(df):
    X = df.drop(columns=["Price"])
    y = df["Price"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"\n  Train set : {X_train.shape[0]} samples")
    print(f"  Test set  : {X_test.shape[0]} samples")
    print(f"  Features  : {list(X.columns)}")

    return X_train, X_test, y_train, y_test, X_train_scaled, X_test_scaled, X.columns


# ─────────────────────────────────────────────
#  3. TRAIN & EVALUATE MODELS
# ─────────────────────────────────────────────

def evaluate_model(name, model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "R²": r2,
        "Predictions": y_pred,
    }


def train_all_models(X_train, X_test, y_train, y_test,
                     X_train_sc, X_test_sc):
    print("\n" + "=" * 60)
    print("  STEP 3: MODEL TRAINING & EVALUATION")
    print("=" * 60)

    results = []

    # Linear Regression (needs scaled data)
    results.append(evaluate_model("Linear Regression",
                                   LinearRegression(),
                                   X_train_sc, X_test_sc, y_train, y_test))

    # Ridge Regression
    results.append(evaluate_model("Ridge Regression",
                                   Ridge(alpha=10),
                                   X_train_sc, X_test_sc, y_train, y_test))

    # Decision Tree
    results.append(evaluate_model("Decision Tree",
                                   DecisionTreeRegressor(max_depth=6, random_state=42),
                                   X_train, X_test, y_train, y_test))

    # Random Forest
    results.append(evaluate_model("Random Forest",
                                   RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
                                   X_train, X_test, y_train, y_test))

    # Gradient Boosting
    results.append(evaluate_model("Gradient Boosting",
                                   GradientBoostingRegressor(n_estimators=100, random_state=42),
                                   X_train, X_test, y_train, y_test))

    print(f"\n  {'Model':<22} {'RMSE':>14} {'MAE':>14} {'R²':>8}")
    print("  " + "-" * 60)
    for r in results:
        print(f"  {r['Model']:<22} ZAR {r['RMSE']:>10,.0f} ZAR {r['MAE']:>10,.0f} {r['R²']:>8.4f}")

    return results


# ─────────────────────────────────────────────
#  4. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_results(results, y_test, feature_names, X_train, y_train):
    best = max(results, key=lambda x: x["R²"])

    # ── Plot 1: Model Comparison ──────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Regression Model Comparison", fontsize=14, fontweight="bold")

    names = [r["Model"] for r in results]
    rmses = [r["RMSE"] for r in results]
    maes = [r["MAE"] for r in results]
    r2s = [r["R²"] for r in results]
    colors = ["#E74C3C" if n == best["Model"] else "#AED6F1" for n in names]

    axes[0].barh(names, rmses, color=colors)
    axes[0].set_title("RMSE (lower = better)")
    axes[0].set_xlabel("ZAR")

    axes[1].barh(names, maes, color=colors)
    axes[1].set_title("MAE (lower = better)")
    axes[1].set_xlabel("ZAR")

    axes[2].barh(names, r2s, color=colors)
    axes[2].set_title("R² Score (higher = better)")
    axes[2].set_xlabel("R²")
    axes[2].axvline(1.0, color="green", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig("plot_01_model_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("\n  Saved: plot_01_model_comparison.png")

    # ── Plot 2: Actual vs Predicted (best model) ──────────────
    fig, ax = plt.subplots(figsize=(8, 7))
    ax.scatter(y_test, best["Predictions"], alpha=0.5, s=30, color="#3498DB")
    lims = [min(y_test.min(), best["Predictions"].min()),
            max(y_test.max(), best["Predictions"].max())]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect Prediction")
    ax.set_xlabel("Actual Price (ZAR)")
    ax.set_ylabel("Predicted Price (ZAR)")
    ax.set_title(f"Actual vs Predicted – {best['Model']}\n(R² = {best['R²']:.4f})")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plot_02_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_02_actual_vs_predicted.png")

    # ── Plot 3: Feature Importance (Random Forest) ────────────
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    importance = pd.Series(rf.feature_importances_, index=feature_names).sort_values()

    fig, ax = plt.subplots(figsize=(8, 5))
    importance.plot(kind="barh", color="#27AE60", ax=ax)
    ax.set_title("Feature Importance – Random Forest")
    ax.set_xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig("plot_03_feature_importance.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_03_feature_importance.png")

    print(f"\n  Best model: {best['Model']}  (R² = {best['R²']:.4f}, RMSE = ZAR {best['RMSE']:,.0f})")


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CODVEDA INTERNSHIP | LEVEL 2 – TASK 1")
    print("  Predictive Modelling – House Price Regression")
    print("=" * 60)

    df = create_dataset()
    print(f"\n  Dataset created: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Price range: ZAR {df['Price'].min():,.0f} – ZAR {df['Price'].max():,.0f}")
    print(f"  Mean price : ZAR {df['Price'].mean():,.0f}")

    X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc, feat_names = prepare_data(df)
    results = train_all_models(X_train, X_test, y_train, y_test, X_tr_sc, X_te_sc)
    plot_results(results, y_test, feat_names, X_train, y_train)

    print("\n  All outputs saved. Regression pipeline complete.")


if __name__ == "__main__":
    main()
