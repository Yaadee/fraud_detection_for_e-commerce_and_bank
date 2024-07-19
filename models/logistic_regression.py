import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from imblearn.over_sampling import SMOTE
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Load the data
data = pd.read_csv('data/processed_data/processed_creditcard.csv')

# Split the data into features and target
X = data.drop('Class', axis=1)
y = data['Class']

# Apply SMOTE to balance the dataset
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Split the resampled data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.3, random_state=42)

# Initialize the logistic regression model
logreg = LogisticRegression(max_iter=1000)

# Train the model
logreg.fit(X_train, y_train)

# Evaluate the model using cross-validation
cv_scores = cross_val_score(logreg, X_res, y_res, cv=3, scoring='roc_auc')

# Print the cross-validation scores
print(f'Cross-validation ROC AUC scores: {cv_scores}')
print(f'Mean cross-validation ROC AUC score: {cv_scores.mean()}')

# Predict on the test set
y_pred = logreg.predict(X_test)

# Calculate and print the ROC AUC score for the test set
roc_auc = roc_auc_score(y_test, y_pred)
print(f'Test ROC AUC score: {roc_auc}')
