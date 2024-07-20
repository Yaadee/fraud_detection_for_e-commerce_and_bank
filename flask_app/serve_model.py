from flask import Flask, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load models
logistic_regression_model = joblib.load('../models/logistic_regression.pkl')
decision_tree_model = joblib.load('../models/decision_tree.pkl')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    df = pd.DataFrame(data)
    
    # Assuming logistic regression model is used for prediction
    predictions = logistic_regression_model.predict(df)
    
    return jsonify({'predictions': predictions.tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
