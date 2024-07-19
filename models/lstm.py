import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import Adam
from sklearn.metrics import classification_report, roc_auc_score

def load_data(file_path):
    return pd.read_csv(file_path)

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, input_shape=input_shape, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(50, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])
    return model

def train_lstm(X, y, epochs=10, batch_size=32):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Standardize the data
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Reshape for LSTM
    X_train = np.reshape(X_train, (X_train.shape[0], 1, X_train.shape[1]))
    X_test = np.reshape(X_test, (X_test.shape[0], 1, X_test.shape[1]))

    # Create and train the LSTM model
    model = create_lstm_model((X_train.shape[1], X_train.shape[2]))
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)

    # Predict and evaluate
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    y_pred_proba = model.predict(X_test)

    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("ROC-AUC Score:", roc_auc_score(y_test, y_pred_proba))

if __name__ == "__main__":
    df_fraud = load_data('data/processed_data/processed_fraud_with_features.csv')
    X_fraud = df_fraud.drop(columns=['class'])
    y_fraud = df_fraud['class']
    train_lstm(X_fraud, y_fraud)

    df_credit = load_data('data/processed_data/processed_creditcard.csv')
    X_credit = df_credit.drop(columns=['Class'])
    y_credit = df_credit['Class']
    train_lstm(X_credit, y_credit)
