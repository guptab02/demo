import streamlit as st
import joblib
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





# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

# Load the ML model
with open("decisiontree_youtubeadview.pkl", "rb") as file:
    model = joblib.load(file)

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
    print(input_data)
	
    #category={'Fitness': 1,'Music':2,'Travel':3,'Electronics':4,'Movie':5,'Fashion':6,'Vlog':7,'Comedy':8}
    #input_data["category"] = input_data["category"].map(category)
	
    # Update the "category" column using np.where
    input_data["category"] = np.where(input_data["category"] == 'Fitness', 1,
                           np.where(input_data["category"] == 'Music', 2,
                           np.where(input_data["category"] == 'Travel', 3,
                           np.where(input_data["category"] == 'Electronics', 4,
                           np.where(input_data["category"] == 'Movie', 5,
                           np.where(input_data["category"] == 'Fashion', 6,
                           np.where(input_data["category"] == 'Vlog', 7,
                           np.where(input_data["category"] == 'Comedy', 8, 0))))))))
    print(input_data)
       
    # Perform prediction
    try:
        prediction = model.predict(input_data)
        st.success(f"Predicted AdViews: {int(prediction[0])}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")

