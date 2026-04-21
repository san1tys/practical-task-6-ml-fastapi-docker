import os

import requests
import streamlit as st

API_URL = os.environ.get("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Wine Classifier", layout="centered")
st.title("Wine Classifier")
st.write("Enter wine chemical features to predict the wine class.")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        alcohol = st.number_input("Alcohol", value=14.23, format="%.2f")
        malic_acid = st.number_input("Malic Acid", value=1.71, format="%.2f")
        ash = st.number_input("Ash", value=2.43, format="%.2f")
        alcalinity_of_ash = st.number_input("Alcalinity of Ash", value=15.60, format="%.2f")
        magnesium = st.number_input("Magnesium", value=127.0, format="%.1f")

    with col2:
        total_phenols = st.number_input("Total Phenols", value=2.80, format="%.2f")
        flavanoids = st.number_input("Flavanoids", value=3.06, format="%.2f")
        nonflavanoid_phenols = st.number_input("Nonflavanoid Phenols", value=0.28, format="%.2f")
        proanthocyanins = st.number_input("Proanthocyanins", value=2.29, format="%.2f")

    with col3:
        color_intensity = st.number_input("Color Intensity", value=5.64, format="%.2f")
        hue = st.number_input("Hue", value=1.04, format="%.2f")
        od280_od315_of_diluted_wines = st.number_input("OD280/OD315", value=3.92, format="%.2f")
        proline = st.number_input("Proline", value=1065.0, format="%.1f")

    submitted = st.form_submit_button("Predict", use_container_width=True)

if submitted:
    payload = {
        "alcohol": alcohol,
        "malic_acid": malic_acid,
        "ash": ash,
        "alcalinity_of_ash": alcalinity_of_ash,
        "magnesium": magnesium,
        "total_phenols": total_phenols,
        "flavanoids": flavanoids,
        "nonflavanoid_phenols": nonflavanoid_phenols,
        "proanthocyanins": proanthocyanins,
        "color_intensity": color_intensity,
        "hue": hue,
        "od280_od315_of_diluted_wines": od280_od315_of_diluted_wines,
        "proline": proline,
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
