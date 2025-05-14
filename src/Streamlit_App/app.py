# app.py
import streamlit as st
import pickle
import pandas as pd
import numpy as np
# Import the function from results.py
from results import get_sepsis_subtype_and_treatment  # Corrected import

# Page Title with Style
st.title("👨‍⚕️ Sepsis Prediction & Sub-Type Classification App")
st.markdown("---")

# Welcome Message with Style
st.write(
    "🩺 **Welcome to the Sepsis Detection & Subtype Classification App!**\n\n"
    "Provide the patient's medical measurements below and click on **'Predict Sepsis'** to get a real-time diagnosis and possible subtype recommendation. Let’s assist in early detection and informed treatment decisions."
)


# Load the model and key components
with open('models/model_and_key_components.pkl', 'rb') as file:
    loaded_components = pickle.load(file)

# Load KMeans model and subtype mapping
with open('models/kmeans_sepsis_model.pkl', 'rb') as file:
    kmeans_loaded_components = pickle.load(file)

loaded_model = loaded_components['model']
loaded_encoder = loaded_components['encoder']
loaded_scaler = loaded_components['scaler']
kmeans = kmeans_loaded_components['kmeans']
subtype_map = kmeans_loaded_components['subtype_map']

# Data Fields
data_fields = {
    "PRG": "Number of pregnancies (applicable only to females)",
    "PL": "Plasma glucose concentration (mg/dL)",
    "PR": "Diastolic blood pressure (mm Hg)",
    "SK": "Triceps skinfold thickness (mm)",
    "TS": "2-hour serum insulin (mu U/ml)",
    "M11": "Body mass index (BMI) (weight in kg / {(height in m)}^2)",
    "BD2": "Diabetes pedigree function (mu U/ml)",
    "Age": "Age of the patient (years)"
}

# About Section with Style
st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app predicts sepsis based on medical input data. "
    "It uses a machine learning model trained on a dataset of sepsis cases."
)

# Sidebar with Data Fields
st.subheader("📝 Enter Patient Data")
input_data = {}

for field, description in data_fields.items():
    label = f"**{field}** – {description}"
    input_data[field] = st.number_input(label, value=0.0)



# Function to preprocess input data
def preprocess_input_data(input_data):
    numerical_cols = ['PRG', 'PL', 'PR', 'SK', 'TS', 'M11', 'BD2', 'Age']
    input_df = pd.DataFrame([input_data], columns=numerical_cols)
    input_data_scaled = loaded_scaler.transform(input_df)
    return pd.DataFrame(input_data_scaled, columns=numerical_cols)

# Function to make predictions
def make_predictions(input_data_scaled_df):
    y_pred = loaded_model.predict(input_data_scaled_df)
    sepsis_mapping = {0: 'Negative', 1: 'Positive'}
    return sepsis_mapping[y_pred[0]]

# Predict Button with Style
if st.button("💡 Predict Sepsis"):
    try:
        if all(value == 0.0 for value in input_data.values()):
            st.warning("Please enter values before predicting and classifying.")
        else:
            input_data_scaled_df = preprocess_input_data(input_data)
            sepsis_status = make_predictions(input_data_scaled_df)

            if sepsis_status == 'Positive':
                st.warning(f"The predicted sepsis status is: {sepsis_status}")

                # Cluster-based Sepsis Subtype Prediction
                cluster = kmeans.predict(input_data_scaled_df.values)[0]
                subtype = subtype_map.get(cluster, "Unknown")
                st.markdown(f"### 🔬 Cluster-Based Subtype: `{subtype}`")

                # Get AI Recommendation
                st.info("🧠 Fetching subtype description and treatment...")
                gpt_response = get_sepsis_subtype_and_treatment(subtype)
                st.markdown("### 🧬 Sepsis Subtype and Suggested Treatment")
                st.markdown(gpt_response)

            else:
                st.success(f"The predicted sepsis status is: {sepsis_status}")

    except Exception as e:
        st.error(f"An error occurred: {e}")


# Display Data Fields and Descriptions
st.sidebar.title("🔍 Data Fields")
for field, description in data_fields.items():
    st.sidebar.text(f"{field}: {description}")

# Footer
st.markdown(
    """
    <hr style="margin-top: 50px; margin-bottom: 10px;">
    <div style="text-align: center;">
        <small>👨‍💻 Developed by <b>Bhavesh</b></small>
    </div>
    """,
    unsafe_allow_html=True
)
# Add a footer with a link to the GitHub repository
st.markdown(
    """
    <div style="text-align: center;">
        <small>🔗 Check out the code on <a href="https://github.com/bhaveshjain2603/Sepsis-Project">GitHub</a></small>
    </div>
    """,
    unsafe_allow_html=True
)