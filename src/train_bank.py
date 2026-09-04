import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# print("=" * 60)
# print(" CHURN PREDICTION SUITE - DATA EXPLORATION")
# print("=" * 60)
# Load Bank dataset
bank_path = "data/bank_churn.csv"
bank_df = pd.read_csv(bank_path)
# print("\n" + "=" * 60)
# print("2. BANK CUSTOMER CHURN DATASET")
# print("=" * 60)
# print(f"   Shape: {bank_df.shape}")
# print(f"   Columns: {bank_df.columns.tolist()}")
# print(f"\n   First 5 rows:")
# print(bank_df.head())
# print(f"\n   Data types:")
# print(bank_df.dtypes)
# print(f"\n   Missing values:")
# print(bank_df.isnull().sum())
# print(f"\n   Target distribution (Exited):")
# print(bank_df['churn'].value_counts())
# print("\n" + "=" * 60)
# print(" Data exploration complete!")
# print("=" * 60)
# --- CLEANING ---
print("\n" + "=" * 60)
print("CLEANING BANK DATA")
print("=" * 60)

# 1. Drop customer_id from features
X = bank_df.drop(['customer_id', 'churn'], axis=1)
y = bank_df['churn']

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# 2. Check missing values
print(f"\n   Missing values:\n{X.isnull().sum()}")

# 3. Identify numeric and categorical columns
numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

print(f"\n   Numeric columns: {numeric_cols}")
print(f"   Categorical columns: {categorical_cols}")
# --- TRAIN/TEST SPLIT ---
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n   Train size: {X_train.shape[0]}")
print(f"   Test size: {X_test.shape[0]}")

# --- PREPROCESSING ---
# Define columns (use the ones you already identified)
numeric_cols = ['credit_score', 'age', 'tenure', 'balance', 'products_number', 
                'credit_card', 'active_member', 'estimated_salary']
categorical_cols = ['country', 'gender']

# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)

# --- PIPELINE ---
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced'))
])

# --- TRAIN ---
pipeline.fit(X_train, y_train)

# --- EVALUATE ---
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\n   Random Forest Accuracy: {accuracy * 100:.2f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred))

# --- SAVE MODEL ---
# joblib.dump(pipeline, 'models/bank_model.joblib')
# print("\n Bank model saved as 'models/bank_model.joblib'")