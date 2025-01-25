import streamlit as st
import joblib
import pandas as pd

# Load the trained model and training columns
model = joblib.load('rf_model.pkl')
df=pd.read_csv('cars_ml.csv')
# Title and description
st.title("Car Price Prediction App")
st.write("""
This app predicts the price of a car based on its features. Fill in the details below to get a prediction.
""")
columns=['Brand','Mileage','Fuel','Transmission','Year','County','Price_per_Mile']
# Input features
Brand = st.selectbox('Brand',df.Brand.unique()) 
Mileage = st.number_input("Mileage (mil)", min_value=0, max_value=500000, value=10000)
Fuel = st.selectbox("Fuel Type", options=["Petrol", "Diesel", "Electric", "Hybrid", "Natural Gas", "Ethanol"])
Transmission = st.selectbox("Transmission", options=["Manual", "Automatic"])
Year = st.number_input("Year of Manufacture", min_value=1980, max_value=2025, value=2020)
County =st.selectbox('County',df.County.unique())
Price_per_Mile = st.number_input("Price per Mile ", min_value=0, max_value=15555, value=55)
# Predict button
if st.button("Predict Price"):
    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'Brand': [Brand],
        'Mileage': [Mileage],
        'Fuel': [Fuel],
        'Transmission': [Transmission],
        'Year': [Year],
        'County': [County],
        'Price per Mile':[Price_per_Mile]
    })

    # One-hot encode the input to match training data
    input_data = pd.get_dummies(input_data)

    # Align input data with the model's training columns
    input_data = input_data.reindex(columns=columns, fill_value=0)

    # Make prediction
    prediction = model.predict(input_data)[0]

    # Display the result
    st.success(f"The predicted price of the car is: SEK {prediction:,.2f}")
