import io
import os
import streamlit as st
import pandas as pd
import joblib

st.write(os.listdir())

MODEL_LOCATION = "models/model.joblib"

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

  y_pred = model.predict(data)

  st.write("Prediction successful! Preview: ")
  y_preview = pd.concat([y_pred.head(), y_pred.tail()], axis=0)
  y_preview = pd.concat([y_preview, data_preview], axis=1)
  st.dataframe(y_preview)




