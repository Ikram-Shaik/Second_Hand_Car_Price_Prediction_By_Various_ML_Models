import streamlit as st
import pandas as pd
import joblib

# --- Model Scores Dictionary (Train, Test R²) ---
model_scores = {
    "Linear Regression": (0.8118, 0.6943),
    "Ridge Regression": (0.8098, 0.6967),
    "Lasso Regression": (0.8116, 0.6825),
    "Elastic Net": (0.7573, 0.6565),
    "Decision Tree": (0.9325, 0.7102),
    "K-Nearest Neighbours": (0.9137, 0.7924),
    "Random Forest": (0.9646, 0.7874),
    "Gradient Boosting": (0.9467, 0.7941),
    "XGBoost": (0.9619, 0.8090),
}

# --- Load Pickled Model ---
def load_model(model_name):
    return joblib.load(f"models/{model_name}.pkl")

# --- Load All Models ---
model_dict = {
    "Linear Regression": load_model("linear_model"),
    "Ridge Regression": load_model("ridge_model"),
    "Lasso Regression": load_model("lasso_model"),
    "Elastic Net": load_model("elasticnet_model"),
    "Decision Tree": load_model("dt_model"),
    "K-Nearest Neighbours": load_model("knn_model"),
    "Random Forest": load_model("rf_model"),
    "Gradient Boosting": load_model("gbr_model"),
    "XGBoost": load_model("xgb_model"),
}

# --- Streamlit Page Setup ---
st.set_page_config(page_title="Second-Hand Car Price Predictor", layout="centered")
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>🚗 Second-Hand Car Price Predictor</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align: center;'>Select a model and enter car details to estimate the selling price.</p>", unsafe_allow_html=True)

# --- Model Selector ---
selected_model_name = st.selectbox("🔍 Choose a Model to Predict", list(model_dict.keys()))
model = model_dict[selected_model_name]
train_score, test_score = model_scores[selected_model_name]

# --- Car Input Form ---
st.header("📝 Enter Car Details")
col1, col2 = st.columns(2)

with col1:
    brand = st.text_input("Brand", "Maruti")
    year = st.number_input("Year of Manufacture", min_value=1990, max_value=2025, value=2015)
    fuel_type = st.selectbox("Fuel Type", ["Petrol", "Diesel", "CNG", "LPG"])

with col2:
    model_name = st.text_input("Model", "Swift")
    km_driven = st.number_input("Kilometers Driven", min_value=0, value=50000)
    transmission = st.selectbox("Transmission", ["Manual", "Automatic"])

mileage = st.slider("Mileage (km/l)", min_value=0.0, max_value=40.0, value=20.0, step=0.1)

# --- Input Preparation ---
input_data = {
    'brand': brand.lower(),
    'model': model_name.lower(),
    'year': year,
    'km_driven': km_driven,
    'fuel_type': fuel_type.lower(),
    'transmission_type': transmission.lower(),
    'mileage': mileage
}
input_df = pd.DataFrame([input_data])

# --- Predict with Selected Model ---
if st.button("🎯 Predict with Selected Model"):
    try:
        predicted_price = int(model.predict(input_df)[0])
        st.success(f"💰 Estimated Selling Price: ₹{predicted_price:,}")

        # --- Show Training and Testing Score Cards ---
        st.markdown("### 📊 Model Performance")
        score1, score2 = st.columns(2)
        score1.markdown(f"""
            <div style='background-color:#E6FFED;padding:20px;border-radius:10px'>
                <h4 style='color:black'>Training R²</h4>
                <h2 style='color:#2E8B57'>{train_score:.4f}</h2>
            </div>
        """, unsafe_allow_html=True)
        score2.markdown(f"""
            <div style='background-color:#E6F0FF;padding:20px;border-radius:10px'>
                <h4 style='color:black'>Testing R²</h4>
                <h2 style='color:#1E90FF'>{test_score:.4f}</h2>
            </div>
        """, unsafe_allow_html=True)

    except Exception as e:
        st.error(f"❌ Prediction Failed: {e}")

# --- Predict with All Models ---
if st.button("📈 Compare Predictions from All Models"):
    result_dict = {}
    for name, mdl in model_dict.items():
        try:
            price = int(mdl.predict(input_df)[0])
            result_dict[name] = f"₹{price:,}"
        except:
            result_dict[name] = "❌ Error"

    st.markdown("### 🧮 Model Predictions (All Models)")

    # Convert to DataFrame
    result_df = pd.DataFrame.from_dict(result_dict, orient='index', columns=["Predicted Price"])
    result_df.index.name = "Model"
    result_df.reset_index(inplace=True)

    # Display with HTML table styling
    st.markdown("""
    <style>
    .big-table {
        background-color: transparent;
    }
    .big-table th {
        font-size: 20px !important;
        text-align: left;
        padding: 10px 20px;
        background-color: #f0f2f6 !important;
        color: black !important;
    }
    .big-table td {
        font-size: 20px !important;
        text-align: left;
        padding: 10px 20px;
        color: white !important;
        background-color: transparent !important;
        border-bottom: 1px solid #444444;
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown(result_df.to_html(classes="big-table", index=False, escape=False), unsafe_allow_html=True)

# Footer watermark (centered)
st.markdown("""
<hr style="margin-top: 50px;"/>
<div style='text-align: center; padding-top: 10px; font-size: 18px; color: gray;'>
    🚗 Built with ❤️ by <strong>Ikram Shaik</strong>
</div>
""", unsafe_allow_html=True)
