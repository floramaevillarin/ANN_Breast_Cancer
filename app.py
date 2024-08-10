import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.datasets import load_breast_cancer

def get_input_data():
    # Load the breast cancer dataset
    data = load_breast_cancer()
    data_df = pd.DataFrame(data.data, columns=data.feature_names)
    data_df["target"] = data.target
    
    # Define input fields with labels and keys
    features = [
        ("Radius (mean)", "mean radius"),
        ("Texture (mean)", "mean texture"),
        ("Perimeter (mean)", "mean perimeter"),
        ("Area (mean)", "mean area"),
        ("Smoothness (mean)", "mean smoothness"),
        ("Compactness (mean)", "mean compactness"),
        ("Concavity (mean)", "mean concavity"),
        ("Concave points (mean)", "mean concave points"),
        ("Symmetry (mean)", "mean symmetry"),
        ("Fractal dimension (mean)", "mean fractal dimension"),
        ("Radius (se)", "radius error"),
        ("Texture (se)", "texture error"),
        ("Perimeter (se)", "perimeter error"),
        ("Area (se)", "area error"),
        ("Smoothness (se)", "smoothness error"),
        ("Compactness (se)", "compactness error"),
        ("Concavity (se)", "concavity error"),
        ("Concave points (se)", "concave points error"),
        ("Symmetry (se)", "symmetry error"),
        ("Fractal dimension (se)", "fractal dimension error"),
        ("Radius (worst)", "worst radius"),
        ("Texture (worst)", "worst texture"),
        ("Perimeter (worst)", "worst perimeter"),
        ("Area (worst)", "worst area"),
        ("Smoothness (worst)", "worst smoothness"),
        ("Compactness (worst)", "worst compactness"),
        ("Concavity (worst)", "worst concavity"),
        ("Concave points (worst)", "worst concave points"),
        ("Symmetry (worst)", "worst symmetry"),
        ("Fractal dimension (worst)", "worst fractal dimension")
    ]

    # Create a dictionary for input fields
    inputs = {}
    for label, feature in features:
        inputs[feature] = st.number_input(
            label,
            min_value=float(0),
            max_value=float(data_df[feature].max()),
            value=float(data_df[feature].mean())
        )
    
    return inputs

def main():
    st.title("Breast Cancer Prediction")
    
    # Load the pre-trained model and scaler
    model = pickle.load(open("best_model.pkl", "rb"))
    scaler = pickle.load(open("scaler.pkl", "rb"))
    
    # Get user input data
    user_inputs = get_input_data()
    
    # Predict based on user input
    if st.button("Predict"):
        input_array = np.array(list(user_inputs.values())).reshape(1, -1)
        scaled_input = scaler.transform(input_array)
        prediction = model.predict(scaled_input)
        
        # Display prediction result
        st.write("Prediction: ", "Malignant" if prediction == 1 else "Benign")
        st.success("Prediction Complete")

if __name__ == '__main__':
    main()
