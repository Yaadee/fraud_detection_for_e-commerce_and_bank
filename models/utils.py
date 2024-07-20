import pandas as pd
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def prepare_data(file_path, target_col):
    data = pd.read_csv(file_path)
    X = data.drop(target_col, axis=1)
    y = data[target_col]
    smote = SMOTE(random_state=42)
    X_res, y_res = smote.fit_resample(X, y)
    X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test
