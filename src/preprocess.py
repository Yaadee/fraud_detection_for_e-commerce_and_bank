# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import LabelEncoder
# from imblearn.over_sampling import SMOTE
# import pandas as pd

# def load_data(file_path):
#     """Load data from a CSV file."""
#     return pd.read_csv(file_path)

# def handle_missing_values(df):
#     """Handle missing values in the dataframe."""
#     for column in df.columns:
#         if df[column].dtype == "object":
#             df[column].fillna(df[column].mode()[0], inplace=True)
#         else:
#             df[column].fillna(df[column].median(), inplace=True)
#     return df

# def clean_data(df):
#     """Remove duplicates from the dataframe."""
#     return df.drop_duplicates()

# def encode_categorical(df):
#     """Encode categorical features using Label Encoding."""
#     le = LabelEncoder()
#     for col in df.select_dtypes(include=['object']).columns:
#         df[col] = le.fit_transform(df[col])
#     return df

# def preprocess_fraud_data():
#     """Preprocess the fraud data."""
#     df = load_data('data/raw_data/Fraud_Data.csv')
#     df = handle_missing_values(df)
#     df = clean_data(df)
#     df = encode_categorical(df)
#     df.to_csv('data/processed_data/processed_Fraud_Data.csv', index=False)

# def preprocess_credit_data():
#     """Preprocess the credit card data."""
#     df = load_data('data/raw_data/creditcard.csv')
#     df = handle_missing_values(df)
#     df.to_csv('data/processed_data/processed_creditcard.csv', index=False)

# def preprocess_ip_data():
#     """Preprocess the IP address to country data."""
#     df = load_data('data/raw_data/IpAddress_to_Country.csv')
#     df = clean_data(df)
#     df = encode_categorical(df)
#     df.to_csv('data/processed_data/processed_ip.csv', index=False)

# def prepare_creditcard_data(creditcard_file):
#     """Prepare the credit card data for training."""
#     creditcard_data = pd.read_csv(creditcard_file)
#     X = creditcard_data.drop(columns=['Class'])
#     y = creditcard_data['Class']
#     return train_test_split(X, y, test_size=0.3, random_state=42)

# def prepare_fraud_data(fraud_file, ip_to_country_file):
#     """Prepare the fraud data for training."""
#     fraud_data = pd.read_csv(fraud_file)
#     ip_to_country = pd.read_csv(ip_to_country_file)
    
#     # Function to merge IP address to country mapping
#     def merge_ip_to_country(ip):
#         try:
#             return ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip) & 
#                                  (ip_to_country['upper_bound_ip_address'] >= ip)]['country'].values[0]
#         except IndexError:
#             return 'Unknown'
    
#     fraud_data['country'] = fraud_data['ip_address'].apply(merge_ip_to_country)
    
#     X = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
#     y = fraud_data['class']
    
#     # Train-test split
#     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
#     # Apply SMOTE to balance the training data
#     smote = SMOTE(random_state=42)
#     X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
#     return X_train_smote, X_test, y_train_smote, y_test

# if __name__ == "__main__":
#     preprocess_fraud_data()
#     preprocess_credit_data()
#     preprocess_ip_data()
#     prepare_creditcard_data()



from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import pandas as pd

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def handle_missing_values(df):
    """Handle missing values in the dataframe."""
    for column in df.columns:
        if df[column].dtype == "object":
            df[column].fillna(df[column].mode()[0], inplace=True)
        else:
            df[column].fillna(df[column].median(), inplace=True)
    return df

def clean_data(df):
    """Remove duplicates from the dataframe."""
    return df.drop_duplicates()

def encode_categorical(df):
    """Encode categorical features using Label Encoding."""
    le = LabelEncoder()
    for col in df.select_dtypes(include=['object']).columns:
        df[col] = le.fit_transform(df[col])
    return df

def preprocess_fraud_data():
    """Preprocess the fraud data."""
    df = load_data('data/raw_data/Fraud_Data.csv')
    df = handle_missing_values(df)
    df = clean_data(df)
    df = encode_categorical(df)
    df.to_csv('data/processed_data/processed_Fraud_Data.csv', index=False)

def preprocess_credit_data():
    """Preprocess the credit card data."""
    df = load_data('data/raw_data/creditcard.csv')
    df = handle_missing_values(df)
    df.to_csv('data/processed_data/processed_creditcard.csv', index=False)

def preprocess_ip_data():
    """Preprocess the IP address to country data."""
    df = load_data('data/raw_data/IpAddress_to_Country.csv')
    df = clean_data(df)
    df = encode_categorical(df)
    df.to_csv('data/processed_data/processed_ip.csv', index=False)

def prepare_creditcard_data(creditcard_file):
    """Prepare the credit card data for training."""
    creditcard_data = pd.read_csv(creditcard_file)
    X = creditcard_data.drop(columns=['Class'])
    y = creditcard_data['Class']
    return train_test_split(X, y, test_size=0.3, random_state=42)

def prepare_fraud_data(fraud_file, ip_to_country_file):
    """Prepare the fraud data for training."""
    fraud_data = pd.read_csv(fraud_file)
    ip_to_country = pd.read_csv(ip_to_country_file)
    
    # Function to merge IP address to country mapping
    def merge_ip_to_country(ip):
        try:
            return ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip) & 
                                 (ip_to_country['upper_bound_ip_address'] >= ip)]['country'].values[0]
        except IndexError:
            return 'Unknown'
    
    fraud_data['country'] = fraud_data['ip_address'].apply(merge_ip_to_country)
    
    X = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
    y = fraud_data['class']
    
    # Save the processed fraud data
    fraud_data.to_csv('data/processed_data/processed_fraud_with_country.csv', index=False)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Apply SMOTE to balance the training data
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    
    return X_train_smote, X_test, y_train_smote, y_test

if __name__ == "__main__":
    preprocess_fraud_data()
    preprocess_credit_data()
    preprocess_ip_data()
    
    # Prepare and save the training and testing data for credit card fraud
    X_train_smote, X_test, y_train_smote, y_test = prepare_fraud_data('data/processed_data/processed_Fraud_Data.csv', 'data/processed_data/processed_ip.csv')
    X_train_smote.to_csv('data/processed_data/X_train_smote.csv', index=False)
    X_test.to_csv('data/processed_data/X_test.csv', index=False)
    y_train_smote.to_csv('data/processed_data/y_train_smote.csv', index=False)
    y_test.to_csv('data/processed_data/y_test.csv', index=False)
