import pandas as pd
import numpy as np
print("=" * 60)
print(" CHURN PREDICTION SUITE - DATA EXPLORATION ")
print("=" * 60)
# Load Telco dataset
telco_path = "data/telco_customer_churn.csv"
telco_df = pd.read_csv(telco_path)
print("\n" + "=" * 60)
print("1. TELCO CUSTOMER CHURN DATASET")
print("=" * 60)
print(f"   Shape: {telco_df.shape}")
print(f"   Columns: {telco_df.columns.tolist()}")
print(f"\n   First 5 rows:")
print(telco_df.head())
print(f"\n   Data types:")
print(telco_df.dtypes)
print(f"\n   Missing values:")
print(telco_df.isnull().sum())
print(f"\n   Target distribution (Churn):")
print(telco_df['Churn'].value_counts())
print("\n" + "=" * 60)
print(" Data exploration complete!")
print("=" * 60)