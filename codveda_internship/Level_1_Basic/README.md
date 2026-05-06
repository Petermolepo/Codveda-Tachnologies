# Level 1 – Basic

## Task 2: Data Cleaning and Preprocessing

**File:** `task2_data_cleaning/data_cleaning.py`

**What it does:**
- Generates a raw messy customer dataset (200 rows)
- Handles missing values (median for numerics, mode for categoricals)
- Removes duplicates
- Fixes inconsistent categorical values (e.g. "male" vs "Male")
- Detects and removes outliers using the IQR method
- Encodes categoricals (one-hot for City, label encoding for Gender)
- Normalises numeric features with StandardScaler

**Outputs:**
- `raw_customer_data.csv` – original raw dataset
- `cleaned_customer_data.csv` – fully cleaned and processed dataset

**Run:**
```bash
python data_cleaning.py
```

---

## Task 3: Exploratory Data Analysis (EDA)

**File:** `task3_eda/eda.py`

**What it does:**
- Generates a retail customer dataset (300 rows)
- Computes summary statistics (mean, median, std, skewness, kurtosis)
- Visualises distributions, categorical features, box plots, scatter plots
- Produces a correlation heatmap
- Group-level analysis by city and gender
- Writes a plain-text insights report

**Outputs:**
- `retail_customers.csv` – dataset
- `plot_01_distributions.png` – histograms
- `plot_02_categorical.png` – bar charts
- `plot_03_boxplots.png` – box plots
- `plot_04_scatter.png` – scatter plots
- `plot_05_correlation.png` – correlation heatmap
- `plot_06_grouped.png` – grouped analysis
- `eda_report.txt` – written insights report

**Run:**
```bash
python eda.py
```
