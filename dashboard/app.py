import json
import joblib
import pandas as pd
import requests
import streamlit as st

st.set_page_config(page_title="Customer purchase predictor", page_icon="🛍️")
st.title("🛍️ Customer Purchase Predictor")
st.markdown(
    "Fill out the customer information below to predict whether they'll make a purchase.")

# Loading expected feature names
try:
    with open("data/outputs/feature_columns.json", encoding="utf-8") as f:
        feature_names = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    st.error("⚠️ Could not load model feature layout.")
    st.stop()
with st.form("predict_form"):
    st.subheader("📋 Customer Information")

    col1, col2 = st.columns(2)
    with col1:
        age = st.slider("Age", 18, 80, 30)
        gender = st.radio("Gender", ["Male", "Female"])
        income = st.number_input(
            "Annual Income($)", 10000, 200000, 65000, step=1000)
        time_spent = st.slider("Time Spent on Website (minutes)", 1, 60, 10)
    with col2:
        purchases = st.slider("Number of Purchases", 0, 20, 3)
        discounts = st.radio("Discounts Availed", [0, 1])
        loyalty = st.radio("Loyalty Program Member", [0, 1])
        product = st.selectbox("Product Category", ["1", "2", "3", "4"])
# Computed feature
    income_per_minute = round(income / time_spent, 2)
    st.metric("🧮 Income per Minute", f"${income_per_minute}")
# Submit button
    submit = st.form_submit_button("🔮 Predict Purchase")
if submit:
    try:
        gender_encoded = 1 if gender == "Female" else 0
        product_onehot = [1 if product == str(i) else 0 for i in range(1, 5)]
        features = [
            age,
            gender_encoded,
            income,
            purchases,
            time_spent,
            discounts,
            income_per_minute,
            *product_onehot,
            loyalty,

        ]
        # Send to FastAPI
        res = requests.post("http://localhost:8000/predict",
                            json={"features": features}, timeout=10)

        response = res.json()
        # Display results interactively
        if "purchase_prediction" in response:
            prediction = response["purchase_prediction"]
            confidence = round(float(response.get("confidence", 0)) * 100, 2)

            st.subheader("🎯 Prediction Result")
            if prediction == 1:
                st.success("🛒 The customer is **likely to purchase**!")
                st.progress(int(confidence))
            else:
                st.warning("🚫 The customer is **unlikely to purchase**.")
                st.progress(int(confidence))

            st.metric(label="Model Confidence",
                      value=f"{confidence}%", delta=None)
        elif "error" in response:
            st.error(f"🚨 API Error: {response['error']}")
        else:
            st.warning("⚠️ Unexpected response from the prediction API.")

    except (requests.exceptions.RequestException, ValueError, KeyError) as e:
        st.exception(f"💥 Something went wrong: {e}")
        st.stop()
    st.success("✅ Prediction completed successfully!")

    st.balloons()  # Celebrate with balloons!
    st.markdown("---")
    st.markdown("### 📊 Feature Importance")
    try:
        # Load the model to get feature importances
        with open("models/random_forest_v1.pkl", "rb") as f:
            model = joblib.load(f)

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({
            "Feature": feature_names,
            "Importance": importances
        }).sort_values(by="Importance", ascending=False)

        st.bar_chart(feature_importance_df.set_index("Feature"))
    except (FileNotFoundError, IOError) as e:
        st.error(f"⚠️ Could not load model for feature importance: {e}")
