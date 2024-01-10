import json
import pandas as pd
from snowflake.snowpark import functions as F
from snowflake.snowpark.functions import pandas_udf
from snowflake.snowpark.session import Session
from snowflake.snowpark.types import *
from snowflake.snowpark.version import VERSION
import streamlit as st
from sklearn.tree import DecisionTreeClassifier

# Read credentials
with open('cred.json') as f:
    connection_parameters = json.load(f)
session = Session.builder.configs(connection_parameters).create()

# Load your datasets
crop_data = session.table('CROP_DATA').toPandas()
products_data = session.table('PRODUCTS').toPandas()

# Select relevant features for the classification task
features = ['Crop_Year', 'Season']  # Include 'Season' as a feature
X = crop_data[features]
y = crop_data['Crop']

# Convert categorical features to numerical using one-hot encoding
X = pd.get_dummies(X)

# Instantiate a decision tree classifier
clf = DecisionTreeClassifier(random_state=42)

# Train the classifier on the entire dataset
clf.fit(X, y)

# Streamlit app
st.title("Products Recommendation")

# Sidebar for user input
st.sidebar.header("Enter Future Date and Season Information")
future_year = st.sidebar.number_input("Enter Future Year:", min_value=2024, max_value=2100, value=2024)
selected_season = st.sidebar.selectbox("Select Season:", crop_data['Season'].unique())

# Create a dataframe for future dates
future = pd.DataFrame(index=pd.date_range(crop_data['Crop_Year'].max(), periods=1))

# Replace placeholder values with actual expected values for the next 90 days
future['Crop_Year'] = future_year
future['Season'] = selected_season

# Convert categorical features to numerical using one-hot encoding
future = pd.get_dummies(future)

# Ensure the features match those used during training
missing_features = set(X.columns) - set(future.columns)
for feature in missing_features:
    future[feature] = 0

# Reorder columns to match the order during training
future = future[X.columns]

# Make predictions for multiple crops and probabilities
predicted_probabilities = clf.predict_proba(future)

# Assuming you want to display the top 5 predicted crops
top_n_predictions = pd.DataFrame({
    'Crop': clf.classes_,
    'Probability': predicted_probabilities[0]
}).nlargest(5, 'Probability')

# Add index to top_n_predictions
top_n_predictions['Index'] = range(1, len(top_n_predictions) + 1)
top_n_predictions = top_n_predictions.set_index('Index')

# Display top N predictions
st.write("Predicted Crops:")
st.write(top_n_predictions)

# Display products information for the top predicted crops
st.write("Products Recommendation for {} Crops:".format(selected_season))
top_crops_info = products_data[products_data['Crop'].isin(top_n_predictions['Crop'])]
st.write(top_crops_info)
