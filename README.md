# 🚗 Second-Hand Car Price Prediction

This project uses multiple machine learning regression models to predict the selling price of used cars based on various features like brand, model, year, kilometers driven, fuel type, transmission type, and mileage.

## 📊 Dataset

- Contains over **21,000 cleaned records**
- Features:
  - `brand`
  - `model`
  - `year`
  - `km_driven`
  - `fuel_type`
  - `transmission_type`
  - `mileage`
  - `selling_price`

---

## 🤖 Machine Learning Models Used

- Linear Regression  
- Ridge Regression  
- Lasso Regression  
- Elastic Net  
- Decision Tree  
- Random Forest  
- K-Nearest Neighbours  
- Gradient Boosting  
- XGBoost  
- Support Vector Regression (SVR)

All models were trained, evaluated (R² Score, RMSE, MAE), and saved using `joblib` or `pickle`.

---

## 🌐 Streamlit Web App

A user-friendly Streamlit interface lets users:
- Choose any trained model
- Enter car details (brand, year, km, etc.)
- Get an instant predicted price
- Compare predictions from all models with training & testing scores

---

## 🚀 How to Run Locally

```bash
# Clone the repository
git clone https://github.com/Ikram-Shaik/Second_Hand_Car_Price_Prediction_By_Various_ML_Models.git
cd Second_Hand_Car_Price_Prediction_By_Various_ML_Models

# Install dependencies
pip install -r requirements.txt

# Launch the Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
├── app.py                  # Streamlit application
├── models/                 # All saved machine learning models
├── requirements.txt        # Project dependencies
└── README.md               # Project description
```

---

## 👨‍💻 Developed By

**Ikram Shaik**  
Built with ❤️ using Python, Scikit-learn, and Streamlit

Check the Running project at : https://huggingface.co/spaces/Ikram-Shaik/Second_Hand_Car_Prediction
