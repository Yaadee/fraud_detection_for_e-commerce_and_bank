import lime
import lime.lime_tabular
import joblib
import pandas as pd
from utils import prepare_data

def explain_model_with_lime(model_path, data_path, target_col):
    # Load model
    model = joblib.load(model_path)
    
    # Prepare data
    X_train, X_test, y_train, y_test = prepare_data(data_path, target_col)
    
    # Initialize LIME explainer
    explainer = lime.lime_tabular.LimeTabularExplainer(
        X_train.values, 
        feature_names=X_train.columns,
        class_names=['Not Fraud', 'Fraud'],
        discretize_continuous=True
    )
    
    # Explain instance
    i = 0  # first instance
    exp = explainer.explain_instance(X_test.iloc[i].values, model.predict_proba, num_features=10)
    exp.show_in_notebook(show_all=False)

if __name__ == "__main__":
    explain_model_with_lime("models/logistic_regression.pkl", "data/processed_data/processed_fraud_with_features.csv", "class")
    explain_model_with_lime("models/logistic_regression.pkl", "data/processed_data/processed_creditcard.csv", "Class")
