import pickle
import numpy as np
import pandas as pd

# Define model paths
model_paths = {
    '1': ('Linear Regression', 'models/linear_regression.pkl'),
    '2': ('Ridge Regression', 'models/ridge_model.pkl'),
    '3': ('Lasso Regression', 'models/lasso_model.pkl'),
    '4': ('Decision Tree', 'models/dt_model.pkl'),
    '5': ('Random Forest', 'models/rf_model.pkl'),
    '6': ('XGBoost', 'models/xgb_model.pkl'),
    '7': ('Gradient Boosting', 'models/gbr_model.pkl'),
    '8': ('Elastic Net', 'models/elasticnet_model.pkl')
}

# Show choices to user
print("\nSelect a model to predict the car price:")
for key, (name, _) in model_paths.items():
    print(f"{key}. {name}")

choice = input("Enter your choice (1-7): ").strip()

if choice not in model_paths:
    print("❌ Invalid choice.")
    exit()

model_name, model_file = model_paths[choice]

# Load model (joblib or pickle based on extension)
print(f"\n📦 Loading {model_name} from {model_file}...")
if model_file.endswith('.joblib'):
    model = joblib.load(model_file)
else:
    with open(model_file, 'rb') as f:
        model = pickle.load(f)

# User input (you can later turn this into argparse or GUI)
print("\n🔢 Enter car details to predict price:")
brand = input("Brand: ")
model_name_input = input("Model: ")
year = int(input("Year: "))
km_driven = int(input("KM Driven: "))
fuel_type = input("Fuel Type (Petrol/Diesel/CNG/LPG): ")
transmission = input("Transmission (Manual/Automatic): ")
mileage = float(input("Mileage (e.g. 21.5): "))

# Format the input sample2

sample = pd.DataFrame([{
    'brand': brand,
    'model': model_name_input,  # string type
    'year': year,
    'km_driven': km_driven,
    'fuel_type': fuel_type,
    'transmission_type': transmission,
    'mileage': mileage
}])


# Predict
predicted_price = model.predict(sample)[0]
print(f"\n💰 Predicted Selling Price: ₹{round(predicted_price):,}")
