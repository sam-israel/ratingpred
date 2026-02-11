#MODEL_LOCATION = "models/model.joblib"
MODEL_LOCATION = "models/model_imputeMissingY_countWord.joblib"
TARGET_COL = "review_scores_rating"
CITY_COL = "city"

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import OrdinalEncoder
from capston_polaris_v4 import clean_airbnb_schema, drop_redunt_cols, transformation

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}




def preprocess(df):
    
    # Correct data types
    clean_airbnb_schema(df, inplace=True)

    # Drop redundant columns
    df = drop_redunt_cols(df)

    # Feature engeneering
    df = transformation (df, y_col = None)

    return df

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
  data = pd.read_csv(uploaded)
  st.write("Data upload successfull. Preview: ")
  data_preview = pd.concat([data.head(), data.tail()], axis=0)
  st.dataframe(data_preview)
  
# If TARGET_COL exists, drop it silently
  if TARGET_COL in data.columns:
    y = data.loc[:,TARGET_COL]
    data = data.drop(columns=[TARGET_COL]) 
  else:
    y = None


  if data.columns.str.contains("_isNA").sum() > 0: # Quick hack to determine if the data was preprocessed according to our magical recipe
    st.write("Data is already preprocessed!")
  else:
    st.write("Preprocessing data...")
    data = preprocess(data)
  
  y_pred = model.predict(data)
  y_pred = pd.Series(y_pred, name = f"{TARGET_COL}_PREDICTED")

  st.write("Prediction successful! ")

  #data_with_pred = pd.concat([y_pred, data], axis=1)
  st.dataframe(y_pred, hide_index=True)

  if y is not None:
    
    
    # Can calculate metrics only in cases where the original Y value was not missing
    y_na_idx = y.isna()
    y = y[~y_na_idx]
    y_pred = y_pred[~y_na_idx]
    

    metric = compute_metrics(y_true = y, y_pred=y_pred)
    
    st.write("Model metrics:")
    st.write(metric)
    
