import pandas as pd
import numpy as np
print("=" * 60)
print(" CHURN PREDICTION SUITE - DATA EXPLORATION")
print("=" * 60)
# Load Bank dataset
bank_path = "data/bank_churn.csv"
bank_df = pd.read_csv(bank_path)
print("\n" + "=" * 60)
print("2. BANK CUSTOMER CHURN DATASET")
print("=" * 60)
print(f"   Shape: {bank_df.shape}")
print(f"   Columns: {bank_df.columns.tolist()}")
print(f"\n   First 5 rows:")
print(bank_df.head())
print(f"\n   Data types:")
print(bank_df.dtypes)
print(f"\n   Missing values:")
print(bank_df.isnull().sum())
print(f"\n   Target distribution (Exited):")
print(bank_df['Exited'].value_counts())

print("\n" + "=" * 60)
print(" Data exploration complete!")
print("=" * 60)