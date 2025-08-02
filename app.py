import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import base64
import os

# --- Streamlit page settings ---
st.set_page_config(page_title="Weather Prediction", layout="centered")

st.markdown("<h1 style='text-align:center; color:#2E86C1;'>☀️ Weather Prediction App</h1>", unsafe_allow_html=True)
st.markdown("This app uses a machine learning model to predict weather based on numeric features like temperature, pressure, etc.")

# --- Load dataset ---
@st.cache_data
def load_data():
    return pd.read_csv("ML_DATASET.csv")

df = load_data()

# --- Identify target column ---
target_col = df.columns[-1]  # assuming last column is the label
st.markdown(f"<small>Detected target column: <b>{target_col}</b></small>", unsafe_allow_html=True)

# --- Input feature selection (only numeric columns) ---
input_features = []
user_input = {}

st.sidebar.header("🛠️ Input Weather Features")

for col in df.drop(columns=[target_col]).columns:
    if pd.api.types.is_numeric_dtype(df[col]):
        input_features.append(col)
        min_val = float(df[col].min())
        max_val = float(df[col].max())
        mean_val = float(df[col].mean())
        user_input[col] = st.sidebar.slider(
            label=col,
            min_value=min_val,
            max_value=max_val,
            value=mean_val
        )
    else:
        st.sidebar.write(f"⏩ Skipped non-numeric column: `{col}`")

input_df = pd.DataFrame([user_input])

# --- Model training ---
X = df[input_features]
y = df[target_col]

model = RandomForestClassifier()
model.fit(X, y)

# --- Prediction ---
prediction = model.predict(input_df)
prediction_proba = model.predict_proba(input_df)

# --- Display output ---
st.subheader("🔍 Predicted Weather:")
st.success(f"{prediction[0]}")

st.subheader("📊 Prediction Probability:")
proba_df = pd.DataFrame(prediction_proba, columns=model.classes_)
st.dataframe(proba_df.style.highlight_max(axis=1, color="lightgreen"))

# --- Footer ---
st.markdown("---")
st.markdown("<small>Made with ❤️ using Streamlit & Scikit-Learn</small>", unsafe_allow_html=True)
