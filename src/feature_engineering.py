import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load data from a CSV file
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to merge IP address to country mapping
def merge_ip_to_country(fraud_file, ip_to_country_file):
    fraud_data = pd.read_csv(fraud_file)
    ip_to_country = pd.read_csv(ip_to_country_file)

    # Ensure IP address columns are integers
    ip_to_country['lower_bound_ip_address'] = ip_to_country['lower_bound_ip_address'].astype(int)
    ip_to_country['upper_bound_ip_address'] = ip_to_country['upper_bound_ip_address'].astype(int)
    fraud_data['ip_address'] = fraud_data['ip_address'].astype(int)

    def map_ip_to_country(ip):
        try:
            return ip_to_country[(ip_to_country['lower_bound_ip_address'] <= ip) & 
                                 (ip_to_country['upper_bound_ip_address'] >= ip)]['country'].values[0]
        except IndexError:
            return 'Unknown'

    fraud_data['country'] = fraud_data['ip_address'].apply(map_ip_to_country)
    
    # Ensure 'country' column is of type string before encoding
    fraud_data['country'] = fraud_data['country'].astype(str)
    
    # Encode country column
    le = LabelEncoder()
    fraud_data['country'] = le.fit_transform(fraud_data['country'])
    
    fraud_data.to_csv('data/processed_data/processed_fraud_with_country.csv', index=False)

    X = fraud_data.drop(columns=['user_id', 'signup_time', 'purchase_time', 'device_id', 'ip_address', 'class'])
    y = fraud_data['class']

    return X, y

# Add time-based features to the dataframe
def add_time_based_features(df):
    df['hour_of_day'] = pd.to_datetime(df['signup_time']).dt.hour
    df['day_of_week'] = pd.to_datetime(df['signup_time']).dt.dayofweek
    return df

def feature_engineer_fraud_data():
    df = load_data('data/processed_data/processed_fraud_with_country.csv')
    df = add_time_based_features(df)
    df.to_csv('data/processed_data/processed_fraud_with_features.csv', index=False)

if __name__ == "__main__":
    X, y = merge_ip_to_country('data/processed_data/processed_Fraud_Data.csv', 'data/processed_data/processed_ip.csv')
    feature_engineer_fraud_data()
