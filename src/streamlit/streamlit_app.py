import streamlit as st
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer

# Load your trained model
model = joblib.load("../../data/06_models/tuned_model.pkl")


# Define a function for data preprocessing
def endpoint_prepare_data(data):
    num_attribs = ["longitude", "latitude", "housing_median_age",
                   "total_rooms", "total_bedrooms", "population",
                   "households", "median_income"]

    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
    ])

    return full_pipeline.fit_transform(data)


# Streamlit application start
st.title('Machine Learning Model Prediction')

# Create input fields
longitude = st.number_input('Longitude')
latitude = st.number_input('Latitude')
housing_median_age = st.number_input('Housing Median Age')
total_rooms = st.number_input('Total Rooms')
total_bedrooms = st.number_input('Total Bedrooms')
population = st.number_input('Population')
households = st.number_input('Households')
median_income = st.number_input('Median Income')

input_data = {
    "longitude": longitude,
    "latitude": latitude,
    "housing_median_age": housing_median_age,
    "total_rooms": total_rooms,
    "total_bedrooms": total_bedrooms,
    "population": population,
    "households": households,
    "median_income": median_income
}


# Function to make predictions
def make_prediction(input_data):
    housing = pd.DataFrame([input_data])
    prepared_housing = endpoint_prepare_data(housing)
    prediction = model.predict(prepared_housing)
    return prediction


# Button to make predictions
if st.button('Predict'):
    prediction = make_prediction(input_data)
    st.write(f'Prediction: {prediction[0]}')

# Run the Streamlit app with: streamlit run your_script_name.py
