#MODEL_LOCATION = "models/model.joblib"
#MODEL_LOCATION = "models/model_imputeMissingY_as4.joblib"
MODEL_LOCATION = "models/random_forest.joblib"
TARGET_COL = "review_scores_rating"

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from capston_polaris_v4 import preprocess
from capston_polaris_v4 import compute_metrics


@st.cache_resource
def load_model():
  return joblib.load(MODEL_LOCATION)

try:
#  global model
  model = load_model()
  st.write("Loaded model successfully")

except Exception as e:
  st.error(f"Loading model failed with error {e}")

st.write("Please upload a CSV file")

uploaded = st.file_uploader(label="Data for predictions", type=["csv"])


if uploaded is not None:
  data = pd.read_csv(uploaded, engine="python")
  st.write("Data upload successfull. Preview: ")
  data_preview = pd.concat([data.head(), data.tail()], axis=0)
  st.dataframe(data_preview)
  
# If TARGET_COL exists, drop it silently
  if TARGET_COL in data.columns:
    y = data.loc[:,TARGET_COL]
    data = data.drop(columns=[TARGET_COL]) 
  else:
    y = None


  if "central" in data.columns: # Quick hack to determine if the data was preprocessed according to our magical recipe
    st.write("Data is already preprocessed!")
  else:
    st.write("Preprocessing data...")
    data = preprocess(data,drop_duplicate_rows=False)
  
  y_pred = model.predict(data)
  y_pred = pd.Series(y_pred, name = f"{TARGET_COL}_PREDICTED")

  st.write("Prediction successful! ")

  #data_with_pred = pd.concat([y_pred, data], axis=1)
  st.dataframe(y_pred, hide_index=True)

  if y is not None:
       
    # Can calculate metrics only in cases where the original Y value was not missing
    y_na_idx = y.isna()
    
    metric = compute_metrics(y_true = y[~y_na_idx], y_pred=y_pred[~y_na_idx])
    
    st.write("Model metrics:")
    st.write(metric)
    
