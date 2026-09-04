# 📊 Churn Prediction Suite

A multi‑domain customer churn prediction web app that helps businesses identify at‑risk customers. Built with Python, Scikit‑Learn, XGBoost, and Streamlit.

---

## 🎯 Project Overview

This project builds and deploys machine learning models for two domains:

- **Telco Churn** – Predicts whether a telecom customer will churn.
- **Bank Churn** – Predicts whether a bank customer will exit.

Users can select a domain, input customer data, and get a churn probability with a recommended retention action.

---

---

## 🛠️ Tech Stack

- **Python** – Core programming language
- **Pandas & NumPy** – Data manipulation
- **Scikit‑Learn** – Model building and preprocessing
- **XGBoost** – Advanced gradient boosting
- **Streamlit** – Web app interface
- **Joblib** – Model serialization
- **Git & GitHub** – Version control

---

## 📊 Datasets

| Domain | Source       | Target Column    |
| :----- | :----------- | :--------------- |
| Telco  | Kaggle (IBM) | `Churn` (Yes/No) |
| Bank   | Kaggle       | `churn` (1/0)    |

---

## 🤖 Model Selection

We compared two models for each dataset:

### Telco Dataset

| Model         | Accuracy   | Recall (Churn) | F1-Score (Churn) | Winner |
| :------------ | :--------- | :------------- | :--------------- | :----- |
| Random Forest | 78.57%     | 65%            | 0.62             | ❌     |
| **XGBoost**   | **79.06%** | **73%**        | **0.65**         | ✅     |

### Bank Dataset

| Model         | Accuracy   | Recall (Churn) | F1-Score (Churn) | Winner |
| :------------ | :--------- | :------------- | :--------------- | :----- |
| Random Forest | 85.10%     | 61%            | 0.62             | ❌     |
| **XGBoost**   | **86.20%** | **65%**        | **0.65**         | ✅     |

### Final Selection

| Dataset   | Model       | Accuracy | Recall (Churn) | F1-Score (Churn) |
| :-------- | :---------- | :------- | :------------- | :--------------- |
| **Telco** | **XGBoost** | 79.06%   | 73%            | 0.65             |
| **Bank**  | **XGBoost** | 86.20%   | 65%            | 0.65             |

**XGBoost was selected for both datasets due to better recall and F1-score, critical for churn prediction.**

---

## 🚀 How to Run Locally

1. **Clone the repository:**

```bash
git clone https://github.com/SharmanKushwaha/Churn_Prediction_Suite.git
cd Churn_Prediction_Suite
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```
