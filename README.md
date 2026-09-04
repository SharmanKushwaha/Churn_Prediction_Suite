# 📊 Churn Prediction Suite

A multi‑domain customer churn prediction web app that helps businesses identify at‑risk customers. Built with Python, Scikit‑Learn, and Streamlit.

## 🎯 Project Overview

This project builds and deploys machine learning models for two domains:

- **Telco Churn** – Predicts whether a telecom customer will churn.
- **Bank Churn** – Predicts whether a bank customer will exit.

Users can select a domain, input customer data, and get a churn probability with a recommended retention action.

## 📁 Project Structure

Churn_Prediction_Suite/
├── data/
│ ├── telco_customer_churn.csv
│ └── bank_churn.csv
├── models/
│ ├── telco_model.joblib
│ └── bank_model.joblib
├── src/
│ ├── train_telco.py
│ ├── train_bank.py
│ └── explore_data.py
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
