import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Salary Predictor", layout="centered")

st.title("💼 Income Prediction App")
st.write("Wondering if someone's income is likely to be above $50K? Fill in the details below and find out!")

# Load model and encoder
model, encoder, feature_names = joblib.load("best_model.pkl")

# Education mapping
education_mapping = {
    "Preschool": 1, "1st-4th": 2, "5th-6th": 3, "7th-8th": 4,
    "9th": 5, "10th": 6, "11th": 7, "12th": 8, "HS-grad": 9,
    "Some-college": 10, "Assoc-voc": 11, "Assoc-acdm": 12,
    "Bachelors": 13, "Masters": 14, "Prof-school": 15, "Doctorate": 16
}

# --- User Inputs ---
st.header("Personal Information")

age = st.slider("Age", 17, 90, 30, help="Select the person's age.")

gender = st.selectbox("Gender", ["Male", "Female"])

race = st.selectbox("Race", ["White", "Black", "Asian-Pac-Islander", "Amer-Indian-Eskimo", "Other"])

education = st.selectbox("Highest Education Level", list(education_mapping.keys()))
education_num = education_mapping[education]  # Auto-mapped internally

marital_status = st.selectbox("Marital Status", [
    "Married-civ-spouse", "Divorced", "Never-married", "Separated", "Widowed", "Married-spouse-absent", "Married-AF-spouse"
])

relationship = st.selectbox("Relationship to Family", [
    "Wife", "Own-child", "Husband", "Not-in-family", "Other-relative", "Unmarried"
])

st.header("Work Information")

workclass = st.selectbox("Work Class", [
    "Private", "Self-emp-not-inc", "Self-emp-inc", "Federal-gov", "Local-gov",
    "State-gov", "Without-pay", "Never-worked"
])

occupation = st.selectbox("Occupation", [
    "Tech-support", "Craft-repair", "Other-service", "Sales", "Exec-managerial",
    "Prof-specialty", "Handlers-cleaners", "Machine-op-inspct", "Adm-clerical",
    "Farming-fishing", "Transport-moving", "Priv-house-serv", "Protective-serv", "Armed-Forces"
])

hours_per_week = st.slider("Weekly Work Hours", 1, 99, 40)

capital_gain = st.number_input("Capital Gain (if any)", min_value=0, max_value=100000, value=0, help="Enter any profit from capital gains.")

capital_loss = st.number_input("Capital Loss (if any)", min_value=0, max_value=5000, value=0, help="Enter any capital losses if applicable.")

native_country = st.selectbox("Country of Origin", [
    "United-States", "Cambodia", "England", "Puerto-Rico", "Canada", "Germany",
    "Outlying-US(Guam-USVI-etc)", "India", "Japan", "Greece", "South", "China",
    "Cuba", "Iran", "Honduras", "Philippines", "Italy", "Poland", "Jamaica",
    "Vietnam", "Mexico", "Portugal", "Ireland", "France", "Dominican-Republic",
    "Laos", "Ecuador", "Taiwan", "Haiti", "Columbia", "Hungary", "Guatemala",
    "Nicaragua", "Scotland", "Thailand", "Yugoslavia", "El-Salvador", "Trinadad&Tobago",
    "Peru", "Hong", "Holand-Netherlands"
])

# For simplicity, set fnlwgt to default (hidden from user to avoid confusion)
fnlwgt = 200000

# --- Prediction ---
if st.button("🔍 Predict Income"):
    input_data = pd.DataFrame({
        "age": [age],
        "workclass": [workclass],
        "fnlwgt": [fnlwgt],
        "education": [education],
        "education-num": [education_num],
        "marital-status": [marital_status],
        "occupation": [occupation],
        "relationship": [relationship],
        "race": [race],
        "gender": [gender],
        "capital-gain": [capital_gain],
        "capital-loss": [capital_loss],
        "hours-per-week": [hours_per_week],
        "native-country": [native_country]
    })

    categorical_cols = input_data.select_dtypes(include='object').columns.tolist()
    numerical_cols = input_data.select_dtypes(exclude='object').columns.tolist()

    encoded_input = encoder.transform(input_data[categorical_cols])
    encoded_df = pd.DataFrame(encoded_input, columns=encoder.get_feature_names_out(categorical_cols))

    final_input = pd.concat([input_data[numerical_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)], axis=1)
    final_input = final_input.reindex(columns=feature_names, fill_value=0)

    prediction = model.predict(final_input)
    result = ">50K" if prediction[0] == 1 else "<=50K"

    st.markdown("---")
    st.success(f"**Predicted Income:** {result}")
