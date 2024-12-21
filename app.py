import streamlit as st
import joblib
import pandas as pd
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings("ignore")

# Load the ML model
model = joblib.load("decisiontree_youtubeadview.pkl")

# Streamlit app title
st.title("ML Model Prediction App")

# Input fields for user inputs
st.header("Enter the Details for Prediction")

# Create input fields (modify these according to your model's features)
views = st.number_input("Number of Views:", min_value=100)
likes = st.number_input("Number of likes:", min_value=100)
dislikes = st.number_input("Number of dislikes:", min_value=0)
comment = st.number_input("Number of Comments:", min_value=0)
duration = st.number_input("Video duration (in seconds):", min_value=1)
category = st.selectbox("Select Category:", ["Comedy", "Fitness", "Music", "Travel", 
                                             "Electronics", "Movie", "Vlog", "Fashion"])

# Prepare input data for prediction
if st.button("Predict AdViews"):
    # Create a dictionary for input data
    input_data = {
        "views": [views],
        "likes": [likes],
        "dislikes": [dislikes],
        "comment": [comment],
        "duration": [duration],
        "category": [category]
    }
    input_data = pd.DataFrame(input_data)

    # Map categories to numeric values
    category_mapping = {
        'Fitness': 1, 'Music': 2, 'Travel': 3, 'Electronics': 4,
        'Movie': 5, 'Fashion': 6, 'Vlog': 7, 'Comedy': 8
    }
    input_data["category"] = input_data["category"].map(category_mapping)

    # Debugging: Inspect input data
    print("Input Data for Prediction:")
    print(input_data)

    # Perform prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted AdViews: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
