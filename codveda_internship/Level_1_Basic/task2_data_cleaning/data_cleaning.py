"""
Codveda Technology Internship
Level 1 - Task 2: Data Cleaning and Preprocessing
Author: Peter Molepo
Description: Clean and preprocess a raw dataset to make it suitable for analysis.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings("ignore")


# ─────────────────────────────────────────────
#  1. CREATE A REALISTIC RAW DATASET
# ─────────────────────────────────────────────

def create_raw_dataset():
    """Generate a raw messy dataset that mimics real-world data issues."""
    np.random.seed(42)
    n = 200

    data = {
        "CustomerID": range(1001, 1001 + n),
        "Age": np.where(
            np.random.rand(n) < 0.08,
            np.nan,
            np.random.randint(18, 75, n).astype(float)
        ),
        "Gender": np.random.choice(["Male", "Female", "male", "FEMALE", None, "Unknown"], n,
                                    p=[0.35, 0.35, 0.10, 0.10, 0.05, 0.05]),
        "AnnualIncome": np.where(
            np.random.rand(n) < 0.07,
            np.nan,
            np.random.normal(55000, 20000, n).astype(float)
        ),
        "SpendingScore": np.where(
            np.random.rand(n) < 0.06,
            np.nan,
            np.random.randint(1, 100, n).astype(float)
        ),
        "City": np.random.choice(
            ["Johannesburg", "Cape Town", "Durban", "Pretoria", "Unknown", None], n,
            p=[0.30, 0.25, 0.20, 0.15, 0.07, 0.03]
        ),
        "Purchases": np.where(
            np.random.rand(n) < 0.05,
            np.nan,
            np.random.poisson(15, n).astype(float)
        ),
    }

    # Inject some outliers
    df = pd.DataFrame(data)
    outlier_idx = np.random.choice(n, 5, replace=False)
    df.loc[outlier_idx, "AnnualIncome"] = np.random.choice([500000, -5000, 999999], 5)
    df.loc[outlier_idx[:3], "Age"] = np.random.choice([150, -10, 200], 3)

    df.to_csv("raw_customer_data.csv", index=False)
    return df


# ─────────────────────────────────────────────
#  2. STEP-BY-STEP CLEANING PIPELINE
# ─────────────────────────────────────────────

def step1_inspect(df):
    print("\n" + "=" * 55)
    print("  STEP 1: INITIAL DATA INSPECTION")
    print("=" * 55)
    print(f"\n  Shape       : {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Data Types:\n{df.dtypes.to_string()}")
    print(f"\n  Missing Values:\n{df.isnull().sum().to_string()}")
    print(f"\n  Duplicate Rows: {df.duplicated().sum()}")
    print(f"\n  Sample (first 5 rows):\n{df.head().to_string()}")


def step2_handle_missing(df):
    print("\n" + "=" * 55)
    print("  STEP 2: HANDLE MISSING VALUES")
    print("=" * 55)

    before = df.isnull().sum().sum()

    # Numeric columns → fill with median (robust to outliers)
    for col in ["Age", "AnnualIncome", "SpendingScore", "Purchases"]:
        median_val = df[col].median()
        missing_count = df[col].isnull().sum()
        df[col].fillna(median_val, inplace=True)
        print(f"  {col:20s}: filled {missing_count} nulls with median ({median_val:.2f})")

    # Categorical columns → fill with mode
    for col in ["Gender", "City"]:
        mode_val = df[col].mode()[0]
        missing_count = df[col].isnull().sum()
        df[col].fillna(mode_val, inplace=True)
        print(f"  {col:20s}: filled {missing_count} nulls with mode ('{mode_val}')")

    after = df.isnull().sum().sum()
    print(f"\n  Missing values reduced: {before} → {after}")
    return df


def step3_handle_duplicates(df):
    print("\n" + "=" * 55)
    print("  STEP 3: REMOVE DUPLICATES")
    print("=" * 55)
    before = len(df)
    df.drop_duplicates(inplace=True)
    after = len(df)
    print(f"  Rows removed: {before - after}")
    print(f"  Rows remaining: {after}")
    return df


def step4_fix_inconsistencies(df):
    print("\n" + "=" * 55)
    print("  STEP 4: FIX INCONSISTENT CATEGORICAL VALUES")
    print("=" * 55)

    # Standardise Gender
    print(f"  Gender unique values before: {df['Gender'].unique()}")
    df["Gender"] = df["Gender"].str.capitalize().str.strip()
    df["Gender"] = df["Gender"].replace({"Unknown": "Other"})
    print(f"  Gender unique values after : {df['Gender'].unique()}")

    # Standardise City
    df["City"] = df["City"].str.title().str.strip()
    df["City"] = df["City"].replace({"Unknown": "Other"})
    print(f"  City unique values after   : {df['City'].unique()}")

    return df


def step5_remove_outliers(df):
    print("\n" + "=" * 55)
    print("  STEP 5: DETECT AND REMOVE OUTLIERS (IQR Method)")
    print("=" * 55)

    before = len(df)
    outlier_cols = ["Age", "AnnualIncome"]

    for col in outlier_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = df[(df[col] < lower) | (df[col] > upper)]
        df = df[(df[col] >= lower) & (df[col] <= upper)]
        print(f"  {col:20s}: removed {len(outliers)} outliers  (bounds: {lower:.1f} – {upper:.1f})")

    print(f"\n  Rows before: {before}  |  Rows after: {len(df)}")
    return df


def step6_encode_categoricals(df):
    print("\n" + "=" * 55)
    print("  STEP 6: ENCODE CATEGORICAL VARIABLES")
    print("=" * 55)

    # One-hot encoding for City (nominal)
    df = pd.get_dummies(df, columns=["City"], prefix="City", drop_first=False)
    print("  City        : one-hot encoded")

    # Label encoding for Gender (binary-ish)
    le = LabelEncoder()
    df["Gender_Encoded"] = le.fit_transform(df["Gender"])
    print(f"  Gender      : label encoded  {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df


def step7_scale_features(df):
    print("\n" + "=" * 55)
    print("  STEP 7: NORMALISE / STANDARDISE NUMERICAL FEATURES")
    print("=" * 55)

    numeric_cols = ["Age", "AnnualIncome", "SpendingScore", "Purchases"]

    # StandardScaler (zero mean, unit variance)
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df[numeric_cols])
    scaled_df = pd.DataFrame(scaled, columns=[c + "_Scaled" for c in numeric_cols],
                              index=df.index)
    df = pd.concat([df, scaled_df], axis=1)
    print(f"  StandardScaler applied to : {numeric_cols}")
    print(f"\n  Scaled stats (mean ≈ 0, std ≈ 1):")
    print(scaled_df.describe().loc[["mean", "std"]].round(3).to_string())

    return df


# ─────────────────────────────────────────────
#  3. MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "=" * 55)
    print("  CODVEDA INTERNSHIP | LEVEL 1 – TASK 2")
    print("  Data Cleaning and Preprocessing")
    print("=" * 55)

    # Generate raw data
    raw_df = create_raw_dataset()
    print(f"\n  Raw dataset generated: {len(raw_df)} rows, {raw_df.shape[1]} columns")

    # Run pipeline
    df = raw_df.copy()
    step1_inspect(df)
    df = step2_handle_missing(df)
    df = step3_handle_duplicates(df)
    df = step4_fix_inconsistencies(df)
    df = step5_remove_outliers(df)
    df = step6_encode_categoricals(df)
    df = step7_scale_features(df)

    # Save clean data
    df.to_csv("cleaned_customer_data.csv", index=False)

    print("\n" + "=" * 55)
    print("  CLEANING COMPLETE")
    print("=" * 55)
    print(f"\n  Final dataset shape  : {df.shape}")
    print(f"  Missing values left  : {df.isnull().sum().sum()}")
    print(f"  Output saved to      : cleaned_customer_data.csv")
    print("\n  Final column list:")
    for col in df.columns:
        print(f"    - {col}")


if __name__ == "__main__":
    main()
