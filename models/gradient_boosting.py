import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(df, target_column):
    # Drop datetime columns
    datetime_cols = ['signup_time', 'purchase_time']
    df = df.drop(columns=datetime_cols, errors='ignore')
    
    # Separate features and target
    X = df.drop(columns=[target_column])
    y = df[target_column]
    
    return X, y

def train_gradient_boosting(X, y):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Create gradient boosting model
    gb = GradientBoostingClassifier()

    # Hyperparameter tuning
    param_grid = {'n_estimators': [100, 200, 500], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 10]}
    grid_search = GridSearchCV(gb, param_grid, cv=5, scoring='roc_auc')
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
    X_fraud, y_fraud = preprocess_data(df_fraud, 'class')
    train_gradient_boosting(X_fraud, y_fraud)

    df_credit = load_data('data/processed_data/processed_creditcard.csv')
    X_credit, y_credit = preprocess_data(df_credit, 'Class')
    train_gradient_boosting(X_credit, y_credit)
