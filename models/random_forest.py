import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score
from utils import prepare_data
import mlflow
import mlflow.sklearn

# Enable MLflow logging
mlflow.start_run()

def train_random_forest(X_train, y_train, X_test, y_test):
    forest = RandomForestClassifier(random_state=42)
    forest.fit(X_train, y_train)

    # Cross-validation
    cv_scores = cross_val_score(forest, X_train, y_train, cv=3, scoring='roc_auc')
    print(f'Cross-validation ROC AUC scores: {cv_scores}')
    print(f'Mean cross-validation ROC AUC score: {cv_scores.mean()}')

    # Test evaluation
    y_pred = forest.predict(X_test)
    roc_auc = roc_auc_score(y_test, y_pred)
    print(f'Test ROC AUC score: {roc_auc}')

    # Log metrics to MLflow
    mlflow.log_metric("cv_mean_roc_auc", cv_scores.mean())
    mlflow.log_metric("test_roc_auc", roc_auc)

    # Save the model
    model_path = "models/random_forest.pkl"
    joblib.dump(forest, model_path)
    mlflow.sklearn.log_model(forest, "model")
    print(f'Model saved as {model_path}')

if __name__ == "__main__":
    # Preprocess and prepare data
    ecommerce_file = 'data/processed_data/processed_fraud_with_features.csv'
    creditcard_file = 'data/processed_data/processed_creditcard.csv'
    X_train, X_test, y_train, y_test = prepare_data(ecommerce_file, 'class')
    train_random_forest(X_train, y_train, X_test, y_test)
    
    X_train, X_test, y_train, y_test = prepare_data(creditcard_file, 'Class')
    train_random_forest(X_train, y_train, X_test, y_test)
    
mlflow.end_run()
