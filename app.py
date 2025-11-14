import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path

st.set_page_config(page_title="Attrition Predictor", layout="centered")

st.title("Attrition Predictor (Random Forest)")
st.markdown("This demo app uses a bundled `model.pkl` (dummy model) and `functioncolumn.json` produced from your `random_forest.py`.")
st.markdown("Replace `model.pkl` with your real trained model for production use.")

# Load artifacts
MODEL_PATH = Path("model.pkl")
FC_PATH = Path("functioncolumn.json")

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_columns():
    with open(FC_PATH, 'r') as f:
        return json.load(f)['columns']

model = load_model()
columns = load_columns()

st.header("Enter feature values")
default_vals = {}
cols = st.columns(2)
inputs = []
for i, feature in enumerate(columns):
    c = cols[i % 2]
    # provide a generic numeric input box for every feature (encoded dummies included)
    # default 0 for dummy variables, 30 for Age-like, 1 for ordinal small values. Keep conservative defaults.
    default = 0
    if 'Age' in feature or 'Years' in feature or 'Distance' in feature or 'Income' in feature or 'Rate' in feature:
        default = 30
    if 'PercentSalaryHike' in feature:
        default = 10
    if 'Gender' in feature or 'OverTime' in feature or 'Attrition' in feature:
        default = 0
    val = c.number_input(label=feature, value=float(default))
    inputs.append(val)

if st.button("Predict"):
    X = pd.DataFrame([inputs], columns=columns)
    try:
        proba = model.predict_proba(X)[0][1]
        pred = model.predict(X)[0]
        st.write(f"Predicted probability of attrition: {proba:.4f}")
        st.write("Predicted class:", "Attrition (1)" if pred==1 else "No Attrition (0)")
    except Exception as e:
        st.error(f"Prediction failed: {e}.\nMake sure the model expects the same number and order of features as in `functioncolumn.json`.")

st.markdown("""---
### Notes
- The included `model.pkl` is a dummy model trained on random data with the same number of features as detected in your `random_forest.py`.  
- Replace `model.pkl` with your actual trained model (serialized with `joblib.dump(model, 'model.pkl')`) when ready.
- Make sure feature order and preprocessing (scaling/encoding) used at training time match what the app sends to the model.
""")
