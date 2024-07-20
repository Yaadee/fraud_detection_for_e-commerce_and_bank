import shap
import joblib
import pandas as pd
import matplotlib.pyplot as plt
from utils import prepare_data

def explain_model_with_shap(model_path, data_path, target_col):
    # Load model
    model = joblib.load(model_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_path, target_col)
    
    # Initialize SHAP explainer
    explainer = shap.Explainer(model, X_test)
    shap_values = explainer(X_test)
    
    # Summary Plot
    shap.summary_plot(shap_values, X_test)
    
    # Force Plot for the first instance
    shap.force_plot(explainer.expected_value, shap_values[0], X_test.iloc[0])
    
    # Dependence Plot for a specific feature
    shap.dependence_plot(0, shap_values, X_test)
    
if __name__ == "__main__":
    explain_model_with_shap("models/logistic_regression.pkl", "data/processed_data/processed_fraud_with_features.csv", "class")
    explain_model_with_shap("models/logistic_regression.pkl", "data/processed_data/processed_creditcard.csv", "Class")
