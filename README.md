# ANN Breast Cancer

https://floramaevillarin-ann-breast-cancer-app-wx6pzh.streamlit.app/

## Overview

This project focuses on building a classification model to predict breast cancer using an Artificial Neural Network (ANN). Additionally, it includes developing an interactive web application using Streamlit, allowing users to interact with the model and predictions.

## Project Steps

### Step 1: Data Preparation

- **Download the Dataset:**
  - Obtain the Breast Cancer dataset from a reliable source via the `sklearn.datasets` module.

- **Prepare the Data:**
  - Write a Python script to load, clean, and preprocess the dataset to ensure it is ready for analysis. This includes handling missing values, encoding categorical variables, and normalizing numerical features.

### Step 2: Feature Selection

- **Select Important Features:**
  - Apply feature selection techniques like `SelectKBest` from `sklearn.feature_selection` to identify and select the most relevant features for the model.

### Step 3: Model Tuning with Grid Search CV

- **Tune the Model:**
  - Set up Grid Search Cross-Validation to optimize the hyperparameters of the ANN model using `MLPClassifier` from `sklearn.neural_network`. This step ensures the best possible performance for the model.

### Step 4: Build the ANN Model

- **Create and Train the ANN Model:**
  - Implement the ANN model using Python. Train the model on the prepared breast cancer dataset and evaluate its performance using appropriate metrics such as accuracy, precision, recall, and F1 score.

### Step 5: Develop a Streamlit App

- **Build the Streamlit App:**
  - Develop a Streamlit app that allows users to:
    - Upload and preprocess the breast cancer dataset.
    - Interact with the trained ANN model.
    - View predictions and performance metrics.
  - Integrate these features to create an intuitive and user-friendly interface.

## Project Structure

The project directory contains the following files:

- **`.gitignore`**           : Specifies files and directories to be ignored by Git
- **`README.md`**            : Project documentation
- **`app.py`**               : Main application script
- **`best_model.pkl`**       : Serialized best model for prediction
- **`breast_cancer.ipynb`**  : Jupyter Notebook with data exploration and model training
- **`requirements.txt`**     : Python package dependencies
- **`scaler.pkl`**           : Serialized scaler used for feature scaling


## Installation

To set up the project, follow these steps:

1. **Clone the Repository:**

   ```bash
   git clone https://github.com/floramaevillarin/ANN_Breast_Cancer.git
