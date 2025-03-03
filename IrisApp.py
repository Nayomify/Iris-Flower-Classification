import streamlit as st
import pickle
import numpy as np
from sklearn.datasets import load_iris

# Load iris dataset target names
iris = load_iris()

# Load model and scaler
with open('iris_model.pkl', 'rb') as file:
    model_data = pickle.load(file)

scaler = model_data['scaler']
model = model_data['model']

# Streamlit UI
st.title("Iris Flower Classification")
st.write("Enter petal and sepal dimensions to predict the species.")

# Input fields
sepal_length = st.number_input("Sepal Length (cm)")
sepal_width = st.number_input("Sepal Width (cm)")
petal_length = st.number_input("Petal Length (cm)")
petal_width = st.number_input("Petal Width (cm)")

# Predict button
if st.button("Predict"):
    input_data = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    input_scaled = scaler.transform(input_data)
    prediction = model.predict(input_scaled)
    predicted_species = iris.target_names[prediction[0]]
    
    st.success(f'The predicted species is: {predicted_species}')