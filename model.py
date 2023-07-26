# For hiding warning message from the output
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.ensemble import BalancedRandomForestClassifier


# Step 1: Load the dataset from a CSV file
df = pd.read_csv('generated.csv')

# Step 2: Split the dataset into features (X) and labels (y)
X = df.drop('Result', axis=1)  # Assuming 'Result' is the column name for the target variable
y = df['Result']

# Step 3: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

categorical_features = ['Severity Id', 'Issue Type Name', 'Threat Class', 'Security Risk', 'Cause', 'Analysis Result']
preprocessor = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), categorical_features)],
                                 remainder='passthrough')
X_train = preprocessor.fit_transform(X_train)
X_test = preprocessor.transform(X_test)

# Step 4: Train multiple classifiers using ensemble methods
n_estimators = 100  # Number of estimators for each ensemble model
random_state = 42  # Random state for reproducibility

# Random Forest Classifier
rf_clf = BalancedRandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
rf_clf.fit(X_train, y_train)

# AdaBoost Classifier
ada_clf = AdaBoostClassifier(n_estimators=n_estimators, random_state=random_state)
ada_clf.fit(X_train, y_train)

# Step 5: Make predictions on the testing set using each model
rf_pred = rf_clf.predict(X_test)
ada_pred = ada_clf.predict(X_test)

# Step 6: Create an ensemble prediction by majority voting
ensemble_pred = []
for rf, ada in zip(rf_pred, ada_pred):
    # Count the votes from each classifier
    votes = {
        'False Negative': 0,
        'False Positive': 0,
        'True Positive': 0,
        'Duplicate': 0
    }
    votes[rf] += 1
    votes[ada] += 1

    # Make the ensemble prediction based on majority voting
    ensemble_pred.append(max(votes, key=votes.get))

# Step 7: Evaluate the ensemble model
report = classification_report(y_test, ensemble_pred)
accuracy = accuracy_score(y_test, ensemble_pred)

print("Classification Report:")
print(report)

print("Accuracy:", accuracy)
