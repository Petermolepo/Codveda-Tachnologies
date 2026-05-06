"""
Codveda Technology Internship
Level 1 - Task 3: Exploratory Data Analysis (EDA)
Author: Peter Molepo
Description: Perform EDA to understand the structure, patterns, and trends in data.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

# ── Consistent plot style ──────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.alpha": 0.4,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
})
PALETTE = "Set2"


# ─────────────────────────────────────────────
#  1. GENERATE DATASET
# ─────────────────────────────────────────────

def create_dataset():
    """Create a rich synthetic retail customer dataset for EDA."""
    np.random.seed(7)
    n = 300

    age = np.random.randint(18, 70, n)
    income = np.random.normal(52000, 18000, n).clip(15000, 120000)
    spending = (
        0.0003 * income
        + np.random.normal(30, 15, n)
    ).clip(1, 99).astype(int)

    gender = np.random.choice(["Male", "Female"], n, p=[0.48, 0.52])
    city = np.random.choice(
        ["Johannesburg", "Cape Town", "Durban", "Pretoria"], n,
        p=[0.35, 0.27, 0.22, 0.16]
    )
    category = np.random.choice(
        ["Electronics", "Clothing", "Groceries", "Home & Garden"], n,
        p=[0.25, 0.30, 0.28, 0.17]
    )
    purchases = np.random.poisson(12, n)
    satisfaction = np.clip(
        np.round(spending / 20 + np.random.normal(0, 0.8, n)), 1, 5
    ).astype(int)

    df = pd.DataFrame({
        "CustomerID": range(1, n + 1),
        "Age": age,
        "Gender": gender,
        "City": city,
        "AnnualIncome": income.round(2),
        "SpendingScore": spending,
        "Purchases": purchases,
        "FavouriteCategory": category,
        "SatisfactionRating": satisfaction,
    })

    df.to_csv("retail_customers.csv", index=False)
    return df


# ─────────────────────────────────────────────
#  2. SUMMARY STATISTICS
# ─────────────────────────────────────────────

def summary_statistics(df):
    print("\n" + "=" * 60)
    print("  STEP 1: SUMMARY STATISTICS")
    print("=" * 60)

    numeric = df.select_dtypes(include=np.number).drop(columns=["CustomerID"])
    stats = numeric.describe().T
    stats["median"] = numeric.median()
    stats["skewness"] = numeric.skew()
    stats["kurtosis"] = numeric.kurt()

    print(stats[["mean", "median", "std", "min", "max", "skewness", "kurtosis"]].round(2).to_string())

    print("\n  Categorical Value Counts:")
    for col in ["Gender", "City", "FavouriteCategory"]:
        print(f"\n  {col}:\n{df[col].value_counts().to_string()}")


# ─────────────────────────────────────────────
#  3. VISUALISATIONS
# ─────────────────────────────────────────────

def plot_distributions(df):
    """Histograms for all numeric columns."""
    numeric_cols = ["Age", "AnnualIncome", "SpendingScore", "Purchases", "SatisfactionRating"]

    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    fig.suptitle("Distribution of Numerical Features", fontsize=15, fontweight="bold", y=1.01)
    axes = axes.flatten()

    for i, col in enumerate(numeric_cols):
        axes[i].hist(df[col], bins=25, color="#4A90D9", edgecolor="white", alpha=0.85)
        axes[i].axvline(df[col].mean(), color="#E74C3C", linestyle="--", linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
        axes[i].axvline(df[col].median(), color="#27AE60", linestyle="--", linewidth=1.5, label=f"Median: {df[col].median():.1f}")
        axes[i].set_title(col)
        axes[i].legend(fontsize=9)

    axes[-1].set_visible(False)
    plt.tight_layout()
    plt.savefig("plot_01_distributions.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_01_distributions.png")


def plot_categorical(df):
    """Bar charts for categorical columns."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Categorical Feature Distributions", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, ["Gender", "City", "FavouriteCategory"]):
        counts = df[col].value_counts()
        bars = ax.bar(counts.index, counts.values,
                      color=sns.color_palette(PALETTE, len(counts)))
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1, str(int(bar.get_height())),
                    ha="center", va="bottom", fontsize=9)
        ax.tick_params(axis="x", rotation=15)

    plt.tight_layout()
    plt.savefig("plot_02_categorical.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_02_categorical.png")


def plot_boxplots(df):
    """Box plots to visualise spread and outliers."""
    numeric_cols = ["Age", "AnnualIncome", "SpendingScore", "Purchases"]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    fig.suptitle("Box Plots – Spread and Outliers", fontsize=15, fontweight="bold")

    for ax, col in zip(axes, numeric_cols):
        ax.boxplot(df[col], patch_artist=True,
                   boxprops=dict(facecolor="#AED6F1", color="#154360"),
                   medianprops=dict(color="#E74C3C", linewidth=2),
                   whiskerprops=dict(color="#154360"),
                   capprops=dict(color="#154360"),
                   flierprops=dict(marker="o", color="#E74C3C", alpha=0.5))
        ax.set_title(col)
        ax.set_xticks([])

    plt.tight_layout()
    plt.savefig("plot_03_boxplots.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_03_boxplots.png")


def plot_scatter(df):
    """Scatter plots to see key relationships."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Scatter Plots – Key Relationships", fontsize=15, fontweight="bold")

    # Income vs Spending Score
    colors = {"Male": "#3498DB", "Female": "#E91E8C"}
    for gender, grp in df.groupby("Gender"):
        axes[0].scatter(grp["AnnualIncome"], grp["SpendingScore"],
                        c=colors[gender], alpha=0.6, label=gender, s=40)
    axes[0].set_xlabel("Annual Income (ZAR)")
    axes[0].set_ylabel("Spending Score")
    axes[0].set_title("Income vs Spending Score")
    axes[0].legend()

    # Age vs Purchases
    axes[1].scatter(df["Age"], df["Purchases"],
                    c=df["SatisfactionRating"], cmap="RdYlGn",
                    alpha=0.7, s=40)
    axes[1].set_xlabel("Age")
    axes[1].set_ylabel("Purchases")
    axes[1].set_title("Age vs Purchases (colour = Satisfaction)")
    sm = plt.cm.ScalarMappable(cmap="RdYlGn", norm=plt.Normalize(1, 5))
    plt.colorbar(sm, ax=axes[1], label="Satisfaction Rating")

    plt.tight_layout()
    plt.savefig("plot_04_scatter.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_04_scatter.png")


def plot_correlation(df):
    """Correlation heatmap."""
    numeric = df.select_dtypes(include=np.number).drop(columns=["CustomerID"])
    corr = numeric.corr()

    fig, ax = plt.subplots(figsize=(9, 7))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, square=True, linewidths=0.5, ax=ax,
                annot_kws={"size": 10})
    ax.set_title("Correlation Matrix – Numerical Features",
                 fontsize=14, fontweight="bold", pad=15)

    plt.tight_layout()
    plt.savefig("plot_05_correlation.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_05_correlation.png")


def plot_grouped(df):
    """Group-level analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Group-Level Analysis", fontsize=15, fontweight="bold")

    # Average spending score by city
    city_spending = df.groupby("City")["SpendingScore"].mean().sort_values(ascending=False)
    axes[0].barh(city_spending.index, city_spending.values,
                 color=sns.color_palette("Set2", len(city_spending)))
    axes[0].set_xlabel("Average Spending Score")
    axes[0].set_title("Avg Spending Score by City")
    for i, v in enumerate(city_spending.values):
        axes[0].text(v + 0.3, i, f"{v:.1f}", va="center")

    # Satisfaction distribution by gender
    for gender, grp in df.groupby("Gender"):
        counts = grp["SatisfactionRating"].value_counts().sort_index()
        axes[1].plot(counts.index, counts.values, marker="o", label=gender, linewidth=2)
    axes[1].set_xlabel("Satisfaction Rating")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Satisfaction Distribution by Gender")
    axes[1].legend()
    axes[1].set_xticks([1, 2, 3, 4, 5])

    plt.tight_layout()
    plt.savefig("plot_06_grouped.png", dpi=150, bbox_inches="tight")
    plt.close()
    print("  Saved: plot_06_grouped.png")


# ─────────────────────────────────────────────
#  4. INSIGHTS REPORT
# ─────────────────────────────────────────────

def generate_report(df):
    numeric = df.select_dtypes(include=np.number).drop(columns=["CustomerID"])
    corr = numeric.corr()
    top_corr = (
        corr.where(~np.tril(np.ones(corr.shape)).astype(bool))
        .stack()
        .abs()
        .sort_values(ascending=False)
        .head(5)
    )

    report = f"""
============================================================
  CODVEDA INTERNSHIP | LEVEL 1 – TASK 3
  Exploratory Data Analysis – Insights Report
============================================================

DATASET OVERVIEW
  Rows            : {len(df)}
  Columns         : {df.shape[1]}
  Numeric features: {len(numeric.columns)}

KEY STATISTICS
  Average Age         : {df['Age'].mean():.1f} years
  Average Income      : ZAR {df['AnnualIncome'].mean():,.0f}
  Average Spending    : {df['SpendingScore'].mean():.1f} / 100
  Average Purchases   : {df['Purchases'].mean():.1f} per customer
  Avg Satisfaction    : {df['SatisfactionRating'].mean():.2f} / 5

TOP CORRELATIONS
{top_corr.to_string()}

KEY INSIGHTS
  1. SpendingScore has a moderate positive correlation with
     AnnualIncome – higher earners tend to spend more.
  2. Satisfaction ratings are higher among customers with
     higher spending scores, suggesting loyalty to spend.
  3. {df['City'].value_counts().index[0]} has the highest customer count ({df['City'].value_counts().iloc[0]} customers).
  4. '{df['FavouriteCategory'].value_counts().index[0]}' is the most popular product category.
  5. Age distribution is roughly uniform (18–70),
     indicating no strong age skew in the customer base.
  6. Purchases follow a Poisson distribution (avg ≈ {df['Purchases'].mean():.1f}),
     typical for count-based retail metrics.

GENERATED PLOTS
  plot_01_distributions.png  – Histograms of all numeric features
  plot_02_categorical.png    – Bar charts for categorical features
  plot_03_boxplots.png       – Box plots showing spread & outliers
  plot_04_scatter.png        – Scatter plots (income vs spending, age vs purchases)
  plot_05_correlation.png    – Correlation heatmap
  plot_06_grouped.png        – Group-level analysis

============================================================
"""
    print(report)
    with open("eda_report.txt", "w") as f:
        f.write(report)
    print("  Report saved: eda_report.txt")


# ─────────────────────────────────────────────
#  5. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 60)
    print("  CODVEDA INTERNSHIP | LEVEL 1 – TASK 3")
    print("  Exploratory Data Analysis (EDA)")
    print("=" * 60)

    df = create_dataset()
    print(f"\n  Dataset loaded: {df.shape[0]} rows × {df.shape[1]} columns\n")

    summary_statistics(df)

    print("\n  Generating visualisations...")
    plot_distributions(df)
    plot_categorical(df)
    plot_boxplots(df)
    plot_scatter(df)
    plot_correlation(df)
    plot_grouped(df)

    generate_report(df)


if __name__ == "__main__":
    main()
