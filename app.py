import streamlit as st
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Load the trained scaler and model
preprocessor = joblib.load('scaler.pkl')
model = joblib.load('rf_model.pkl')

# Load the dataset to get all unique brands and counties
df = pd.read_csv('cars_au_to_ml.csv')

# Extract unique brands, models, and counties
unique_brands = df['Brand'].unique()
unique_models = df['Brand_Model'].unique()
unique_counties = df['County'].unique()

# Define the input fields for the user to provide data
st.title("Used Car Price Prediction App")

st.write("Please provide the following details to predict the car price:")

brand = st.selectbox("Brand", unique_brands)
brand_model = st.selectbox("Brand Model", unique_models)
fuel = st.selectbox("Fuel Type", ["Petrol", "Diesel", "Hybrid", "Electric"])
transmission = st.selectbox("Transmission", ["Manual", "Automatic"])
mileage = st.number_input("Mileage", min_value=0, value=10000, step=1000)
year = st.number_input("Year", min_value=1990, max_value=2025, value=2020, step=1)
county = st.selectbox("County", unique_counties)


# Create a DataFrame from the user inputs
input_data = pd.DataFrame({
    'Brand': [brand],
    'Brand_Model': [brand_model],
    'Fuel': [fuel],
    'Transmission': [transmission],
    'Mileage': [mileage],
    'Year': [year],
    'County': [county]
})

# Add a predict button
if st.button('Predict'):
    # Preprocess the input data using the same pipeline used during training
    preprocessed_data = preprocessor.transform(input_data)
    
    # Make predictions using the loaded model
    predictions = model.predict(preprocessed_data)
    
    # Display the prediction
    st.write(f"The predicted price of the car is: SEK {predictions[0]:,.2f}")
