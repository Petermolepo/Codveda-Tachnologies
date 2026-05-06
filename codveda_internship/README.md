# Codveda Technologies – Data Science Internship

**Author:** Peter Molepo  
**Domain:** Data Science  
**Duration:** 1 Month  
**Status:** In Progress (Level 1 Basic Completed)  

---

## 📋 Project Overview

This repository contains the complete codebase for the Codveda Technologies Data Science Internship program. The internship is structured into three levels of increasing complexity:

- **Level 1 (Basic):** Data manipulation, cleaning, and exploratory analysis
- **Level 2 (Intermediate):** Machine learning algorithms and predictive modeling
- **Level 3 (Advanced):** Deep learning and natural language processing

---

## 📁 Project Structure

```
codveda_internship/
│
├── Level_1_Basic/
│   ├── README.md
│   ├── task2_data_cleaning/
│   │   ├── README.md
│   │   ├── data_cleaning.py
│   │   ├── raw_customer_data.csv
│   │   └── cleaned_customer_data.csv
│   └── task3_eda/
│       ├── eda.py
│       ├── retail_customers.csv
│       ├── eda_report.txt
│       └── plot_*.png (6 visualization files)
│
├── Level_2_Intermediate/
│   ├── README.md
│   ├── task1_regression/
│   │   ├── regression.py
│   │   ├── house_prices.csv
│   │   └── plot_*.png (3 visualization files)
│   └── task2_classification/
│       ├── classification.py
│       ├── customer_churn.csv
│       └── plot_*.png (3 visualization files)
│
└── Level_3_Advanced/
    ├── README.md
    ├── task2_nlp/
    │   ├── nlp_classification.py
    │   ├── customer_reviews.csv
    │   └── plot_*.png (3 visualization files)
    └── task3_neural_network/
        └── neural_network.py
```

---

## ✅ Tasks Completed

### Level 1 – Basic

| Task | Status | Description |
|------|--------|-------------|
| Task 1 | ⏳ Pending | Basic Python Programming |
| Task 2 | ✅ Completed | Data Cleaning and Preprocessing |
| Task 3 | ✅ Completed | Exploratory Data Analysis (EDA) |

#### Task 2: Data Cleaning and Preprocessing
- Generated synthetic customer dataset (200 rows, 7 columns)
- Handled missing values using median/mode imputation
- Removed duplicates and outliers (IQR method)
- Encoded categorical variables (one-hot and label encoding)
- Standardized numerical features
- **Output:** `cleaned_customer_data.csv` (160 rows, 16 columns)

#### Task 3: Exploratory Data Analysis
- Analyzed retail customer dataset (300 rows, 9 columns)
- Computed comprehensive statistics (mean, median, std, skewness, kurtosis)
- Generated 6 visualization plots (distributions, correlations, etc.)
- Identified key insights and correlations
- **Output:** `eda_report.txt` and PNG visualization files

### Level 2 – Intermediate
| Task | Status | Description |
|------|--------|-------------|
| Task 1 | ⏳ Pending | Predictive Modelling – Regression |
| Task 2 | ⏳ Pending | Classification with Logistic Regression |

### Level 3 – Advanced
| Task | Status | Description |
|------|--------|-------------|
| Task 1 | ⏳ Pending | Time Series Analysis |
| Task 2 | ⏳ Pending | NLP – Text Classification |
| Task 3 | ⏳ Pending | Neural Networks with TensorFlow/Keras |

---

## 🛠️ Installation & Requirements

### Prerequisites
- Python 3.8+
- Git

### Dependencies
```bash
pip install pandas numpy scikit-learn matplotlib seaborn nltk tensorflow
```

### Individual Task Dependencies
- **Level 1:** `pandas`, `scikit-learn`
- **Level 2:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`
- **Level 3:** `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `seaborn`, `nltk`, `tensorflow`

---

## 🚀 How to Run

### Level 1 Tasks
```bash
# Task 2: Data Cleaning
cd Level_1_Basic/task2_data_cleaning
python data_cleaning.py

# Task 3: EDA
cd Level_1_Basic/task3_eda
python eda.py
```

### Future Tasks
```bash
# Level 2
cd Level_2_Intermediate/task1_regression
python regression.py

cd Level_2_Intermediate/task2_classification
python classification.py

# Level 3
cd Level_3_Advanced/task2_nlp
python nlp_classification.py

cd Level_3_Advanced/task3_neural_network
python neural_network.py
```

---

## 📊 Key Results (Completed Tasks)

### Data Cleaning Insights
- **Dataset Reduction:** 200 → 160 rows (20% reduction from outlier removal)
- **Missing Values:** All handled appropriately (median for numeric, mode for categorical)
- **Feature Engineering:** Added scaled features and encoded categoricals

### EDA Insights
- **Customer Demographics:** Age 18-70, uniform distribution
- **Top Correlations:** SpendingScore vs SatisfactionRating (0.58)
- **Popular Categories:** Clothing (97 customers), Johannesburg (111 customers)
- **Average Metrics:** Age: 44.2, Income: ZAR 50,524, Spending Score: 44.0

---

## 📝 Notes

- All datasets are synthetically generated for demonstration purposes
- Visualizations are saved as PNG files in respective task directories
- Reports are generated in plain text format for easy reading
- Code is well-documented with inline comments

---

## 🔄 Next Steps

1. Complete remaining Level 1 tasks
2. Move to Level 2: Regression and Classification modeling
3. Advance to Level 3: NLP and Neural Networks
4. Final project documentation and presentation

---

**Last Updated:** May 6, 2026  
**Repository:** [GitHub](https://github.com/Petermolepo/Codveda-Tachnologies.git)
```

---

## How to Run

```bash
# Level 1
cd Level_1_Basic/task2_data_cleaning && python data_cleaning.py
cd Level_1_Basic/task3_eda && python eda.py

# Level 2
cd Level_2_Intermediate/task1_regression && python regression.py
cd Level_2_Intermediate/task2_classification && python classification.py

# Level 3
cd Level_3_Advanced/task2_nlp && python nlp_classification.py
cd Level_3_Advanced/task3_neural_network && python neural_network.py
```

---

## Tools & Libraries

- **Python 3.x**
- **pandas, numpy** – data manipulation
- **scikit-learn** – ML models, preprocessing, evaluation
- **matplotlib, seaborn** – visualisation
- **nltk** – NLP preprocessing
- **TensorFlow / Keras** – deep learning

---

## GitHub
[github.com/Petermolepo](https://github.com/Petermolepo)
