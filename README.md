# Customer Churn Prediction

An end-to-end machine learning project to explore customer churn patterns and predict which customers are more likely to leave a telecommunications service.

> **Project status:** Initial reproducible baseline. Model results are generated when the notebook is run; no performance scores are claimed in advance.

## Objectives

- Explore customer demographics, account details, services, and churn patterns.
- Prepare a reproducible machine-learning workflow.
- Compare Logistic Regression and Random Forest classifiers.
- Evaluate performance with accuracy, precision, recall, F1-score, ROC-AUC, and a confusion matrix.
- Explain which variables contribute most to predictions.
- Discuss practical limitations and responsible use of churn predictions.

## Dataset

This project uses the **IBM Telco Customer Churn** sample dataset. IBM describes the data as sample data for a fictional telecommunications company.

- Source repository: [IBM/telco-customer-churn-on-icp4d](https://github.com/IBM/telco-customer-churn-on-icp4d) (archived by IBM in July 2024).
- Direct CSV: [Telco-Customer-Churn.csv](https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv).
- Dataset documentation: [IBM Cognos Analytics sample data](https://www.ibm.com/docs/en/cognos-analytics/11.1.0?topic=samples-telco-customer-churn).

The notebook downloads the CSV into `data/Telco-Customer-Churn.csv` if it is not already present. The CSV is not committed to this repository. Review the source repository's terms and attribution requirements before redistribution.

## Repository structure

```text
customer-churn-prediction/
├── data/
│   └── README.md
├── notebooks/
│   └── customer_churn_analysis.ipynb
├── src/
│   └── train.py
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

## Getting started

### 1. Clone the repository

```bash
git clone https://github.com/protik-deb/customer-churn-prediction.git
cd customer-churn-prediction
```

### 2. Create an environment

```bash
python -m venv .venv
```

Activate it:

- Windows PowerShell: `.venv\Scripts\Activate.ps1`
- macOS/Linux: `source .venv/bin/activate`

### 3. Install dependencies

```bash
python -m pip install -r requirements.txt
```

### 4. Run the notebook

```bash
jupyter lab
```

Open `notebooks/customer_churn_analysis.ipynb` and run all cells. The notebook downloads the dataset on first use, so an internet connection is needed unless the CSV is already present locally.

### 5. Run the training script

The script expects the CSV at `data/Telco-Customer-Churn.csv` by default. Download it manually first when running the script without running the notebook:

**Windows PowerShell**
```powershell
New-Item -ItemType Directory -Force data
Invoke-WebRequest -Uri "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -OutFile "data/Telco-Customer-Churn.csv"
```

**macOS/Linux**
```bash
mkdir -p data
curl -L "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv" -o data/Telco-Customer-Churn.csv
```

Then run:

```bash
python src/train.py
```

Or provide a custom path:

```bash
python src/train.py --data path/to/Telco-Customer-Churn.csv
```

The script prints evaluation metrics and saves the fitted model pipeline to `models/best_model.joblib`.

## Methodology

1. **Data understanding:** inspect dimensions, types, missing values, duplicates, and the target distribution.
2. **Exploratory analysis:** examine churn against contract type, tenure, monthly charges, and service choices.
3. **Preprocessing:** handle blank total charges, encode categorical features, and scale numeric variables within a scikit-learn pipeline.
4. **Validation:** use a stratified train/test split.
5. **Models:** compare Logistic Regression and Random Forest using the same training/test split.
6. **Evaluation:** report class-sensitive metrics and ROC-AUC; inspect the confusion matrix and feature importance where available.

All preprocessing is fitted on training data only through a pipeline to reduce data leakage. The customer identifier is excluded from predictors.

## Model evaluation

The notebook and script calculate the metrics from the selected dataset and print them at runtime. Results are not embedded here as fixed claims because they depend on the dataset version, split, and library versions.

For a more robust assessment, consider repeated stratified cross-validation, threshold tuning based on business costs, calibration checks, and external validation.

## Important limitations

- The dataset represents a fictional telecom provider and may not represent real customer behavior.
- Churn prediction identifies statistical associations, not causes of churn.
- Class imbalance means accuracy alone is not sufficient.
- Feature importance is model-dependent and should not be interpreted as causal evidence.
- Predictions should support human review and service improvement, not be used as the sole basis for adverse customer decisions.

## Tech stack

Python · Pandas · NumPy · Matplotlib · Seaborn · Scikit-learn · Jupyter

## Author

**Protik Deb**  
M.Sc. Applied Mathematics for Networks and Data Sciences  
Hochschule Mittweida, Germany

- GitHub: [protik-deb](https://github.com/protik-deb)
- LinkedIn: [protik-deb-6389181aa](https://www.linkedin.com/in/protik-deb-6389181aa)

---

*An educational portfolio project in data analysis and applied machine learning.*
