import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def load_data(file_path):
    return pd.read_csv(file_path)

def train_mlp(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create MLP model
    mlp = MLPClassifier(max_iter=1000)

    # Hyperparameter tuning
    param_grid = {'hidden_layer_sizes': [(50,), (100,), (50, 50)], 'alpha': [0.0001, 0.001, 0.01]}
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc')
    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    # Predict and evaluate
    y_pred = best_model.predict(X_test)
    y_pred_proba = best_model.predict_proba(X_test)[:, 1]

    print("Best Parameters:", grid_search.best_params_)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

if __name__ == "__main__":
    df_fraud = load_data('data/processed_data/processed_fraud_with_features.csv')
    X_fraud = df_fraud.drop(columns=['class'])
    y_fraud = df_fraud['class']
    train_mlp(X_fraud, y_fraud)

    df_credit = load_data('data/processed_data/processed_creditcard.csv')
    X_credit = df_credit.drop(columns=['Class'])
    y_credit = df_credit['Class']
    train_mlp(X_credit, y_credit)
