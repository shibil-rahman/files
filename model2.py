import pandas as pd
import joblib
from datetime import datetime

# Load the new dataset and preprocess it
new_data = pd.read_csv("sast-deployment_test_input.csv")
new_data.fillna('', inplace=True)
X_new_text = new_data.drop(columns=['Unnamed: 0', 'Severity Id'])  # Exclude the output columns if present
X_new_numerical = new_data[['Severity Id']]

# Load the previously fitted encoder and transform the numerical features of the new data
encoder = joblib.load("encoder.pkl")
X_new_numerical_encoded = encoder.transform(X_new_numerical)

# Load the previously fitted TF-IDF vectorizer and transform the text features of the new data
tfidf_vectorizer = joblib.load("tfidf_vectorizer.pkl")
X_new_text_tfidf = tfidf_vectorizer.transform(X_new_text['Issue Type Name'] + ' ' + X_new_text['Cause'] + ' ' + X_new_text['Threat Class'] + ' ' + X_new_text['Security Risk'])

# Concatenate text and numerical features for the new data
X_new_combined = pd.concat([pd.DataFrame(X_new_numerical_encoded.toarray()), pd.DataFrame(X_new_text_tfidf.toarray())], axis=1)

# Load the previously trained classifier
multioutput_classifier_tuned = joblib.load("multioutput_classifier_tuned.pkl")

# Use the already trained classifier to predict the outputs for the new data
y_new_pred = multioutput_classifier_tuned.predict(X_new_combined)

# Get the predictions for Analysis Result and Result columns
y_new_analysis_result_pred = y_new_pred[:, 0]
y_new_result_pred = y_new_pred[:, 1]

# Add the predictions to the new_data DataFrame
new_data['Analysis Result Prediction'] = y_new_analysis_result_pred
new_data['Result Prediction'] = y_new_result_pred

# Save the DataFrame with the new 'Result' column to a new CSV file
current_date = datetime.now().strftime("%Y%m%d")
output_file_path = f'output_data_{current_date}.csv'
new_data.to_csv(output_file_path, index=False)

print(f"Predictions saved to '{output_file_path}'.")
