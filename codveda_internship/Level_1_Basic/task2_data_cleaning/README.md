# Task 2: Data Cleaning and Preprocessing

## Overview
This task demonstrates comprehensive data cleaning and preprocessing techniques on a synthetic customer dataset. The script generates a raw dataset with intentional imperfections and systematically cleans it through multiple steps.

## Files
- `data_cleaning.py` - Main Python script for data cleaning
- `raw_customer_data.csv` - Generated raw dataset (200 rows, 7 columns)
- `cleaned_customer_data.csv` - Final cleaned and processed dataset (160 rows, 16 columns)

## Data Cleaning Steps

### 1. Initial Data Inspection
- Load and examine dataset structure
- Check data types and missing values
- Identify duplicates

### 2. Handle Missing Values
- **Numerical features**: Fill with median values
  - Age: median = 49.00
  - AnnualIncome: median = 57197.06
  - SpendingScore: median = 48.50
  - Purchases: median = 15.00
- **Categorical features**: Fill with mode values
  - Gender: mode = 'Female'
  - City: mode = 'Cape Town'

### 3. Remove Duplicates
- No duplicates found in this dataset

### 4. Fix Inconsistent Categorical Values
- **Gender**: Standardize to 'Male', 'Female', 'Other'
- **City**: Standardize city names

### 5. Outlier Detection and Removal
- Use IQR (Interquartile Range) method
- **Age**: Removed 3 outliers (bounds: -1.5 – 98.5)
- **AnnualIncome**: Removed 4 outliers (bounds: -2441.8 – 115565.8)
- Dataset reduced from 200 to 160 rows

### 6. Encode Categorical Variables
- **City**: One-hot encoding (creates 5 new columns)
- **Gender**: Label encoding (0=Female, 1=Male, 2=Other, 3=NaN)

### 7. Feature Scaling
- Apply StandardScaler to numerical features
- Features scaled: Age, AnnualIncome, SpendingScore, Purchases
- Result: mean ≈ 0, std ≈ 1 for all scaled features

## Final Dataset
- **Shape**: 160 rows × 16 columns
- **Columns**:
  - CustomerID (original)
  - Age, AnnualIncome, SpendingScore, Purchases (original + scaled versions)
  - Gender (original + encoded)
  - City (one-hot encoded columns)

## Dependencies
- pandas
- scikit-learn

## Usage
```bash
python data_cleaning.py
```

## Output
The script provides detailed console output showing each step's progress and results. The cleaned dataset is saved as `cleaned_customer_data.csv` for use in subsequent analysis tasks.