import streamlit as st
import pandas as pd
import joblib
import os

# Load preprocessing pipeline
preprocessing_pipeline = joblib.load('data/05_model_input/preprocessing_pipeline.joblib')

# Initialize the current model
current_model_name = "best_model.pkl"
model_path = os.path.join("data/06_models", current_model_name)
model = joblib.load(model_path)

models_dir = "data/06_models/"

def list_models():
    return [file for file in os.listdir(models_dir) if file.endswith('.pkl') and file != '.gitkeep']

def select_model(model_name):
    global model, current_model_name
    model_path = os.path.join(models_dir, model_name)
    model = joblib.load(model_path)
    current_model_name = model_name

# Streamlit layout
st.title("California Housing Price Prediction")

# Model selection
model_list = list_models()
selected_model = st.selectbox("Select a model:", model_list, index=model_list.index(current_model_name))
if st.button("Load Selected Model"):
    select_model(selected_model)
    st.success(f"Model changed to {selected_model}")

# Prediction form
with st.form(key='predict_form'):
    st.write("Enter the data for prediction:")
    longitude = st.number_input("Longitude")
    latitude = st.number_input("Latitude")
    housing_median_age = st.number_input("Housing Median Age")
    total_rooms = st.number_input("Total Rooms")
    total_bedrooms = st.number_input("Total Bedrooms")
    population = st.number_input("Population")
    households = st.number_input("Households")
    median_income = st.number_input("Median Income")

    submit_button = st.form_submit_button(label='Predict')

    if submit_button:
        # Prepare data for prediction
        feature_names = ["longitude", "latitude", "housing_median_age", "total_rooms",
                         "total_bedrooms", "population", "households", "median_income"]
        data = [[longitude, latitude, housing_median_age, total_rooms, total_bedrooms,
                 population, households, median_income]]
        input_df = pd.DataFrame(data, columns=feature_names)
        prepared_data = preprocessing_pipeline.transform(input_df)

        # Make a prediction
        prediction = model.predict(prepared_data)
        st.write(f"Predicted House Price: {prediction[0]}")
