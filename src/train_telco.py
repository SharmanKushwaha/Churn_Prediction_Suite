import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
# print("=" * 60)
# print(" CHURN PREDICTION SUITE - DATA EXPLORATION ")
# print("=" * 60) 
# Load Telco dataset
telco_path = "data/telco_customer_churn.csv"
telco_df = pd.read_csv(telco_path)
# print("\n" + "=" * 60)
# print("1. TELCO CUSTOMER CHURN DATASET")
# print("=" * 60)
# print(f"   Shape: {telco_df.shape}")
# print(f"   Columns: {telco_df.columns.tolist()}")
# print(f"\n   First 5 rows:")
# print(telco_df.head())
# print(f"\n   Data types:")
# print(telco_df.dtypes)
# print(f"\n   Missing values:")
# print(telco_df.isnull().sum())
# print(f"\n   Target distribution (Churn):")
# print(telco_df['Churn'].value_counts())
# print("\n" + "=" * 60)
# print(" Data exploration complete!")
# print("=" * 60)
# --- CLEANING ---
print("\n" + "=" * 60)
print("CLEANING TELCO DATA")
print("=" * 60)

# 1. Drop customerID from features
X = telco_df.drop(['customerID', 'Churn'], axis=1)
y = telco_df['Churn']
# Convert target to numeric
y = y.map({'No': 0, 'Yes': 1})

print(f"   Features shape: {X.shape}")
print(f"   Target shape: {y.shape}")

# 2. Convert TotalCharges to numeric (it's currently a string)
print("\n   Converting TotalCharges to numeric...")
X['TotalCharges'] = pd.to_numeric(X['TotalCharges'], errors='coerce')

# Check if any values became NaN after conversion
print(f"   Missing values in TotalCharges after conversion: {X['TotalCharges'].isnull().sum()}")

# Fill missing TotalCharges with median
X['TotalCharges'] = X['TotalCharges'].fillna(X['TotalCharges'].median())
print(f"   Missing values after fill: {X['TotalCharges'].isnull().sum()}")

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
numeric_cols = ['SeniorCitizen', 'tenure', 'MonthlyCharges', 'TotalCharges']
categorical_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                    'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                    'TechSupport', 'StreamingTV', 'StreamingMovies', 'Contract',
                    'PaperlessBilling', 'PaymentMethod']
# Preprocessor
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_cols),
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_cols)
    ]
)
# --- RANDOMFORESTCLASSIFIER PIPELINE ---
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

# --- XGBOOST PIPELINE ---
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=42,
        scale_pos_weight=2  # Helps with class imbalance
    ))
])

# Train
xgb_pipeline.fit(X_train, y_train)

# Evaluate
y_pred_xgb = xgb_pipeline.predict(X_test)
accuracy_xgb = accuracy_score(y_test, y_pred_xgb)
print("\n" + "=" * 60)
print("XGBOOST RESULTS")
print("=" * 60)
print(f"   Accuracy: {accuracy_xgb * 100:.2f}%")
print("\n   Classification Report:")
print(classification_report(y_test, y_pred_xgb))
# --- SAVE MODEL ---
# Compare and save the better model
if accuracy_xgb > accuracy:
    joblib.dump(xgb_pipeline, 'models/telco_model.joblib')
    print("\n XGBoost model saved (better than Random Forest)")
else:
    joblib.dump(pipeline, 'models/telco_model.joblib')
    print("\n Random Forest model saved (better than XGBoost)")