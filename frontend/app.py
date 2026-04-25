import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Wine Quality Predictor", layout="centered")
st.title("Wine Quality Predictor")
st.write(
    "Enter wine physicochemical features to predict quality: "
    "**quality_low** (3–5), **quality_medium** (6), **quality_high** (7–8)."
)

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        fixed_acidity = st.number_input("Fixed Acidity", value=7.4, format="%.2f")
        volatile_acidity = st.number_input("Volatile Acidity", value=0.70, format="%.2f")
        citric_acid = st.number_input("Citric Acid", value=0.0, format="%.2f")
        residual_sugar = st.number_input("Residual Sugar", value=1.9, format="%.2f")

    with col2:
        chlorides = st.number_input("Chlorides", value=0.076, format="%.3f")
        free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11.0, format="%.1f")
        total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34.0, format="%.1f")
        density = st.number_input("Density", value=0.9978, format="%.4f")

    with col3:
        pH = st.number_input("pH", value=3.51, format="%.2f")
        sulphates = st.number_input("Sulphates", value=0.56, format="%.2f")
        alcohol = st.number_input("Alcohol", value=9.4, format="%.2f")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    payload = {
        "fixed_acidity": fixed_acidity,
        "volatile_acidity": volatile_acidity,
        "citric_acid": citric_acid,
        "residual_sugar": residual_sugar,
        "chlorides": chlorides,
        "free_sulfur_dioxide": free_sulfur_dioxide,
        "total_sulfur_dioxide": total_sulfur_dioxide,
        "density": density,
        "pH": pH,
        "sulphates": sulphates,
        "alcohol": alcohol,
    }

    try:
        response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        response.raise_for_status()
        result = response.json()

        st.success(
            f"Predicted class: **{result['predicted_label']}** (index {result['predicted_class']})"
        )

        st.subheader("Class Probabilities")
        probs = result["probabilities"]
        st.bar_chart(probs)
        for label, prob in probs.items():
            st.write(f"- {label}: {prob:.4f}")

    except requests.exceptions.ConnectionError:
        st.error(f"Could not connect to the API at {API_URL}. Is the API service running?")
    except requests.exceptions.Timeout:
        st.error("The API request timed out after 10 seconds.")
    except requests.exceptions.HTTPError as exc:
        st.error(f"API returned an error: {exc.response.status_code} — {exc.response.text}")
    except Exception as exc:
        st.error(f"Unexpected error: {exc}")
