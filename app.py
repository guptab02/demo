import streamlit as st
import pickle
import pandas as pd
import numpy as np
import datetime
import sklearn
import time
from sklearn.model_selection import train_test_split
from sklearn import metrics

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import pickle





# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load the ML model
with open("decisiontree_youtubeadview.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app title
st.title("ML Model Prediction App")

# Input fields for user inputs
st.header("Enter the Details for Prediction")

# Create input fields (modify these according to your model's features)

views = st.number_input("Number of Views:", min_value=100)
likes = st.number_input("Number of likes:", min_value=100)
dislikes = st.number_input("Number of dislikes:", min_value=0)
comment = st.number_input("Number of Comments:", min_value=0)
duration = st.number_input("Video duration:", min_value=1)
category = st.selectbox("Select Category:", ["Comedy", "Fitness", "Music", "Travel", "Electronics", "Movie", "Vlog", "Fashion"])


# Prepare input data for prediction
if st.button("Predict AdViews"):      
    # Create a dataframe with input features    
    input_data = {
    "views": [views],
    "likes": [likes],
    "dislikes": [dislikes],
    "comment": [comment],
    "duration" : [duration],
    "category" : [category]
	}
    input_data = pd.DataFrame([input_data])
       
    # Perform prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted AdViews: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

