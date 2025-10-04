import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ------------------------------
# Load Model from Hugging Face Hub
# ------------------------------
REPO_ID = "vamsikrishna1516/tourism_model"
MODEL_FILENAME = "best_tourism_model_v1.joblib"

st.title("Wellness Tourism Package Purchase Prediction App")
st.write("""
This application predicts whether a customer is likely to purchase the **Wellness Tourism Package**
based on their demographic and interaction details.
""")

# Download model (cached automatically by huggingface_hub)
@st.cache_resource
def load_model():
    model_path = hf_hub_download(repo_id=REPO_ID, filename=MODEL_FILENAME)
    return joblib.load(model_path)

model = load_model()
st.success("Model loaded successfully from Hugging Face Hub!")

# ------------------------------
# User Input Section
# ------------------------------
st.header("Enter Customer Details")

# Numeric inputs
age = st.number_input("Age", min_value=18, max_value=80, value=30)
monthly_income = st.number_input("Monthly Income (₹)", min_value=10000, max_value=500000, value=60000, step=1000)
number_of_person_visiting = st.number_input("Number of Persons Visiting", min_value=1, max_value=10, value=2)
number_of_children = st.number_input("Number of Children Visiting", min_value=0, max_value=5, value=0)
preferred_property_star = st.selectbox("Preferred Property Star", [3, 4, 5])
number_of_trips = st.number_input("Number of Trips (per year)", min_value=0, max_value=50, value=2)
pitch_score = st.slider("Pitch Satisfaction Score", 1, 5, 3)
number_of_followups = st.number_input("Number of Follow-ups", min_value=0, max_value=10, value=2)
duration_of_pitch = st.number_input("Duration of Pitch (mins)", min_value=1, max_value=100, value=15)

# Categorical inputs
typeof_contact = st.selectbox("Type of Contact", ["Company Invited", "Self Inquiry"])
city_tier = st.selectbox("City Tier", [1, 2, 3])
occupation = st.selectbox("Occupation", ["Salaried", "Small Business", "Large Business", "Free Lancer"])
gender = st.radio("Gender", ["Male", "Female"])
marital_status = st.selectbox("Marital Status", ["Single", "Married", "Divorced"])
passport = st.selectbox("Has Passport?", ["No", "Yes"])
own_car = st.selectbox("Owns a Car?", ["No", "Yes"])
designation = st.selectbox("Designation", ["Executive", "Senior Executive", "Manager", "AVP", "VP"])
product_pitched = st.selectbox("Product Pitched", ["Basic", "Standard", "Deluxe", "Super Deluxe", "King"])

# ------------------------------
# Prepare Data for Prediction
# ------------------------------
input_data = pd.DataFrame([{
    "Age": age,
    "TypeofContact": typeof_contact,
    "CityTier": city_tier,
    "Occupation": occupation,
    "Gender": gender,
    "NumberOfPersonVisiting": number_of_person_visiting,
    "PreferredPropertyStar": preferred_property_star,
    "MaritalStatus": marital_status,
    "NumberOfTrips": number_of_trips,
    "Passport": 1 if passport == "Yes" else 0,
    "OwnCar": 1 if own_car == "Yes" else 0,
    "NumberOfChildrenVisiting": number_of_children,
    "Designation": designation,
    "MonthlyIncome": monthly_income,
    "PitchSatisfactionScore": pitch_score,
    "ProductPitched": product_pitched,
    "NumberOfFollowups": number_of_followups,
    "DurationOfPitch": duration_of_pitch
}])

st.subheader("Input Summary")
st.dataframe(input_data)

# ------------------------------
# Prediction Button
# ------------------------------
if st.button("Predict Purchase Probability"):
    try:
        prediction = model.predict(input_data)[0]
        if prediction == 1:
            st.success("The model predicts: **Customer will purchase the Wellness Package.**")
        else:
            st.warning("The model predicts: **Customer will not purchase the Wellness Package.**")
    except Exception as e:
        st.error(f"Prediction failed due to: {e}")
