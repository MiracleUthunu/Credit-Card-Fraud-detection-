# 🛡️ Credit Card Fraud Prediction

A machine learning pipeline that detects fraudulent credit card transactions using **Logistic Regression** and **Random Forest Classification**, built with scikit-learn and trained on real-world transaction data.

---

## 📌 Problem Statement

Credit card fraud causes billions of dollars in losses annually. This project builds a binary classification model to distinguish legitimate transactions from fraudulent ones, helping financial institutions flag suspicious activity in real time.

---

## 📂 Dataset

**File:** `fraud test.csv`

The dataset contains credit card transaction records with the following key features:

| Feature | Description |
|---|---|
| `merchant` | Merchant name |
| `category` | Transaction category |
| `amt` | Transaction amount |
| `dob` | Cardholder date of birth |
| `job` | Cardholder occupation |
| `trans_date_trans_time` | Transaction timestamp |
| `trans_num` | Unique transaction ID |
| `lat`, `long` | Cardholder location |
| `merch_lat`, `merch_long` | Merchant location |
| `is_fraud` | **Target** — 1 = Fraud, 0 = Legitimate |

---

## ⚙️ Project Workflow

```
Raw Data → EDA → Feature Engineering → Model Training → Evaluation → Export
```

### 1. Exploratory Data Analysis (EDA)
- Distribution histograms for all numeric features
- Pairplot to visualize feature correlations
- `.info()` and `.describe()` summaries

### 2. Feature Engineering
- Label-encoded categorical columns: `merchant`, `category`, `dob`, `job`, `trans_date_trans_time`, `trans_num`
- Dropped non-predictive or redundant columns: `state`, `street`, `city`, `gender`, `first`, `last`, `unix_time`, `merch_lat`, `merch_long`

### 3. Model Training
Two models were trained on a 70/30 train-test split (`random_state=40`):

| Model | Key Parameters |
|---|---|
| Logistic Regression | Default scikit-learn settings |
| Random Forest Classifier | `n_estimators=100`, `random_state=42` |

### 4. Evaluation Metrics
- Accuracy
- Precision
- Confusion Matrix
- Mean Absolute Error (MAE)

### 5. Model Export
The final Random Forest model is serialized using `pickle`:
```python
pkl.dump(rfc, open('Fraud test_Model.pkl', 'wb'))
```

---

## 🚀 Getting Started

### Prerequisites

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### Run the Notebook

```bash
jupyter notebook Fraud_Prediction.ipynb
```

### Load the Saved Model

```python
import pickle as pkl

model = pkl.load(open('Fraud test_Model.pkl', 'rb'))
predictions = model.predict(X_new)
```

---

## 📊 Results

| Model | Accuracy | Precision | MAE |
|---|---|---|---|
| Logistic Regression | *See notebook* | *See notebook* | *See notebook* |
| Random Forest | *See notebook* | *See notebook* | *See notebook* |

> **Note:** Random Forest was selected as the final model and exported for deployment.

---

## 🗂️ Repository Structure

```
├── Fraud_Prediction.ipynb     # Main notebook
├── Fraud test_Model.pkl       # Trained Random Forest model
├── fraud test.csv             # Dataset (not included — see note below)
└── README.md
```

> ⚠️ The dataset `fraud test.csv` is not included in this repository due to size constraints. You can source a similar dataset from [Kaggle's Credit Card Fraud Detection datasets](https://www.kaggle.com/search?q=credit+card+fraud).

---

## 🔮 Future Improvements

- [ ] Handle class imbalance with SMOTE or class weighting
- [ ] Hyperparameter tuning with GridSearchCV
- [ ] Add XGBoost / LightGBM models for comparison
- [ ] Build a REST API endpoint using FastAPI or Flask
- [ ] Add cross-validation for more robust evaluation

---

## 🧰 Tech Stack

![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-ML-orange?logo=scikit-learn)
![Pandas](https://img.shields.io/badge/Pandas-Data-green?logo=pandas)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-red?logo=jupyter)

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).
