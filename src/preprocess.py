# from sklearn.preprocessing import LabelEncoder
# import pandas as pd

# def load_data(file_path):
#     # Load data from a CSV file
#     return pd.read_csv(file_path)

# def handle_missing_values(df):
#     # Handle missing values in the dataframe
#     for column in df.columns:
#         if df[column].dtype == "object":
#             df[column].fillna(df[column].mode()[0], inplace=True)
#         else:
#             df[column].fillna(df[column].median(), inplace=True)
#     return df

# def clean_data(df):
#     # Remove duplicates from the dataframe
#     return df.drop_duplicates()

# def encode_categorical(df):
#     # Encode categorical features using Label Encoding
#     le = LabelEncoder()
#     for col in df.select_dtypes(include=['object']).columns:
#         if col not in ['purchase_time', 'signup_time']:  # Exclude encoding purchase_time and signup_time
#             df[col] = le.fit_transform(df[col])
#     return df

# def preprocess_fraud_data():
#     # Preprocess the fraud data
#     df = load_data('data/raw_data/Fraud_Data.csv')
#     df = handle_missing_values(df)
#     df = clean_data(df)
#     df = encode_categorical(df)
#     df.to_csv('data/processed_data/processed_Fraud_Data.csv', index=False)

# def preprocess_credit_data():
#     # Preprocess the credit card data
#     df = load_data('data/raw_data/creditcard.csv')
#     df = handle_missing_values(df)
#     df.to_csv('data/processed_data/processed_creditcard.csv', index=False)

# def preprocess_ip_data():
#     # Preprocess the IP address to country data
#     df = load_data('data/raw_data/IpAddress_to_Country.csv')
#     df = clean_data(df)
#     df = encode_categorical(df)
#     df.to_csv('data/processed_data/processed_ip.csv', index=False)

# if __name__ == "__main__":
#     preprocess_fraud_data()
#     preprocess_credit_data()
#     preprocess_ip_data()



import pandas as pd
from sklearn.model_selection import train_test_split

def load_data(file_path):
    return pd.read_csv(file_path)

def prepare_data(fraud_file, credit_file):
    # Load datasets
    fraud_data = load_data(fraud_file)
    credit_data = load_data(credit_file)
    
    # Split fraud data into features and target
    X_fraud = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
    y_fraud = fraud_data['class']
    
    # Split credit data into features and target
    X_credit = credit_data.drop(columns=['Time', 'Class'])
    y_credit = credit_data['Class']
    
    # Split into train and test sets
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test = train_test_split(X_fraud, y_fraud, test_size=0.2, random_state=42)
    X_credit_train, X_credit_test, y_credit_train, y_credit_test = train_test_split(X_credit, y_credit, test_size=0.2, random_state=42)
    
    return X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test

if __name__ == "__main__":
    fraud_file = 'data/processed_data/processed_fraud_with_features.csv'
    credit_file = 'data/processed_data/processed_creditcard.csv'
    X_fraud_train, X_fraud_test, y_fraud_train, y_fraud_test, X_credit_train, X_credit_test, y_credit_train, y_credit_test = prepare_data(fraud_file, credit_file)



