MODEL_LOCATION = "models/model.joblib"
TARGET_COL = "review_scores_rating"

import io
import os
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae = float(mean_absolute_error(y_true, y_pred))
    r2 = float(r2_score(y_true, y_pred))
    return {"rmse": rmse, "mae": mae, "r2": r2, "n": int(len(y_true))}

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

  y_pred = model.predict(data)
  y_pred = pd.Series(y_pred, name = f"{TARGET_COL}_PREDICTED")

  st.write("Prediction successful! ")
#  y_preview = pd.concat([y_pred.head(), y_pred.tail()], axis=0)
#  y_preview = pd.concat([y_preview, data_preview], axis=1)

  data_with_pred = pd.concat([y_pred, data], axis=1)
  st.dataframe(data_with_pred, hide_index=True)

  if y is not None:
    metric = compute_metrics(y_true = y, y_pred=y_pred)
    
    st.write("Model metrics:")
    st.write(metric)
    
    naive_y_pred = np.full_like(y_pred, fill_value = 4.75)
    naive_metric = compute_metrics(y_true = y, y_pred = naive_y_pred)
    st.write("Naive model metrics:")
    st.write(naive_metric)
