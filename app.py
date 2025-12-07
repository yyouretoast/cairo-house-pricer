import streamlit as st
import joblib
import numpy as np

model = joblib.load('cairo_house_model.pkl')

st.title("🏠 Cairo House Price Predictor")
st.write("Enter the details of the apartment to estimate its price.")

area = st.number_input("Area (in square meters)", min_value=50, max_value=1000, value=120)
bedrooms = st.slider("Number of Bedrooms", min_value=1, max_value=10, value=3)

location_option = st.selectbox(
    "Location",
    ("Nasr City", "Maadi", "New Cairo")
)

location_mapping = {"Nasr City": 0, "Maadi": 1, "New Cairo": 2}
location_code = location_mapping[location_option]

# predict
if st.button("Predict Price"):

    features = np.array([[area, bedrooms, location_code]])
    
    prediction = model.predict(features)[0]
    # display in egp
    st.success(f"Estimated Price: **{prediction:,.0f} EGP**")