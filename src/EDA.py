import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def load_data(file_path):
    """Load data from a CSV file."""
    return pd.read_csv(file_path)

def univariate_analysis(df, column):
    """Perform univariate analysis on a column."""
    sns.histplot(df[column], kde=True)
    plt.title(f'Univariate Analysis of {column}')
    plt.show()

def bivariate_analysis(df, column1, column2):
    """Perform bivariate analysis between two columns."""
    sns.scatterplot(x=df[column1], y=df[column2])
    plt.title(f'Bivariate Analysis between {column1} and {column2}')
    plt.show()

def explore_data():
    fraud_data = load_data('data/processed_data/processed_Fraud_Data.csv')
    credit_data = load_data('data/processed_data/processed_creditcard.csv')

    # Univariate Analysis
    for column in fraud_data.columns:
        univariate_analysis(fraud_data, column)

    for column in credit_data.columns:
        univariate_analysis(credit_data, column)

    # Bivariate Analysis
    bivariate_analysis(fraud_data, 'purchase_value', 'class')
    bivariate_analysis(credit_data, 'V1', 'Class')

if __name__ == "__main__":
    explore_data()
