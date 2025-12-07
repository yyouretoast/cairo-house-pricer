import streamlit as st
import joblib
import numpy as np

# setup
st.set_page_config(page_title="Cairo Real Estate Estimator", page_icon="🏠")

st.title("🏠 Cairo House Price Predictor")
st.write("Enter the property details below to get a price estimate.")

# load artifacts
@st.cache_resource
def load_artifacts():
    model = joblib.load('cairo_house_model.pkl')
    loc_encoder = joblib.load('location_encoder.pkl')
    type_encoder = joblib.load('type_encoder.pkl')
    return model, loc_encoder, type_encoder

try:
    model, loc_encoder, type_encoder = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Artifacts not found! Did you run train_model.py?")
    st.stop()

# input
col1, col2 = st.columns(2)

with col1:
    # Area (Sqm)
    size_sqm = st.number_input("Area (in sqm)", min_value=30, max_value=2000, value=120)
    # Bedrooms
    bedrooms = st.slider("Bedrooms", min_value=1, max_value=10, value=3)

with col2:
    # Bathrooms
    bathrooms = st.slider("Bathrooms", min_value=1, max_value=8, value=2)
    # Property Type

    prop_types = type_encoder.classes_
    selected_type = st.selectbox("Property Type", prop_types)

# Location

locations = loc_encoder.classes_
selected_location = st.selectbox("Location", locations)

# prediction
if st.button("Predict Price 🚀", type="primary"):
    

    loc_encoded = loc_encoder.transform([selected_location])[0]
    type_encoded = type_encoder.transform([selected_type])[0]
    

    features = np.array([[size_sqm, bedrooms, bathrooms, loc_encoded, type_encoded]])
    
    # run
    prediction = model.predict(features)[0]
    
    # result
    st.markdown("---")
    st.subheader(f"💰 Estimated Price: {prediction:,.0f} EGP")
    
    # price per meter
    price_per_meter = prediction / size_sqm
    st.caption(f"That is approximately {price_per_meter:,.0f} EGP per sqm.")