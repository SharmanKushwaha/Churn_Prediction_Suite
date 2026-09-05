import streamlit as st
import pandas as pd
import joblib
# Page Configuration
st.set_page_config(
    page_title="Churn Prediction Suite",
    layout="wide"
)
# Title and description
st.title("Churn Prediction Suite")
st.markdown("Predict customer churn for **Telco** and **Banking** domains.")
# Sidebar for domain selection
st.sidebar.header("Select Domain")
domain = st.sidebar.radio(
    "Choose a domain: ",
    ("Telco", "Bank")
)
st.sidebar.markdown("---")
st.sidebar.info(
    "Fill in the customer details and click **Predict** to get the churn probability."
)
@st.cache_resource
def load_model(domain):
    if domain == "Telco":
        return joblib.load('models/telco_model.joblib')
    else:
        return joblib.load('models/bank_model.joblib')

model = load_model(domain)
# TELCO DOMAIN
if domain == "Telco":
    st.header("Telco Churn Prediction")
    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No", "Yes", "No phone service"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])

    with col2:
        online_security = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        device_protection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
        payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, max_value=200.0, value=50.0)
        total_charges = st.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=500.0)

     # Convert inputs to DataFrame
    input_data = pd.DataFrame([{
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }])

# BANK DOMAIN
else:
    st.header("Bank Customer Churn Prediction")
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=850, value=600)
        country = st.selectbox("Country", ["France", "Spain", "Germany"])
        gender = st.selectbox("Gender", ["Female", "Male"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (years)", min_value=0, max_value=10, value=3)

    with col2:
        balance = st.number_input("Balance ($)", min_value=0.0, max_value=250000.0, value=50000.0)
        products_number = st.selectbox("Number of Products", [1, 2, 3, 4])
        credit_card = st.selectbox("Has Credit Card", [0, 1])
        active_member = st.selectbox("Active Member", [0, 1])
        estimated_salary = st.number_input("Estimated Salary ($)", min_value=0.0, max_value=200000.0, value=50000.0)

    # Convert inputs to DataFrame
    input_data = pd.DataFrame([{
        'credit_score': credit_score,
        'country': country,
        'gender': gender,
        'age': age,
        'tenure': tenure,
        'balance': balance,
        'products_number': products_number,
        'credit_card': credit_card,
        'active_member': active_member,
        'estimated_salary': estimated_salary
    }])

# PREDICT
if st.button(" Predict Churn"):
    try:
        # Make prediction
        prediction = model.predict(input_data)[0]
        probability = model.predict_proba(input_data)[0][1]

         # Determine risk category
        if probability < 0.3:
            risk = " Low"
            action = "No immediate action needed."
        elif probability < 0.6:
            risk = " Medium"
            action = "Send targeted email with tips/offers."
        else:
            risk = " High"
            action = "Offer 1-month discount / call from retention team."

        # Display results
        st.markdown("---")
        st.subheader(" Prediction Result")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")
        with col2:
            st.metric("Risk Category", risk)
        with col3:
            st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")

        st.info(f" **Recommended Action:** {action}")
        
        # Show input data (for transparency)
        with st.expander(" View Input Data"):
            st.dataframe(input_data)
            
    except Exception as e:
        st.error(f"An error occurred: {e}")
        st.info("Please check your inputs and try again.")

# FOOTER
st.markdown("---")
st.markdown(
    "Built with using Streamlit, Scikit-Learn, and XGBoost"
)