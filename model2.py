import pandas as pd
import joblib
from datetime import datetime
import csv
import os
import json
import re
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import RobustScaler


def keep_rows_between(input_file, output_file, start_value, end_value):
    with open(input_file, 'r', newline='') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)
        
        rows_to_keep = []
        keep_mode = False
        
        for row in reader:
            if start_value in row:
                keep_mode = True
            if keep_mode and (start_value not in row and end_value not in row):
                rows_to_keep.append(row)
            if end_value in row:
                keep_mode = False
        
        writer.writerows(rows_to_keep)
        
input_file_path = 'Report_Static_jazzcalm_workspace_2023-02-17_04-22-33_2023-03-02-2.csv'
output_file_path = 'new_main.csv'

start_value = 'Issue Attributes:'
end_value = 'Fix Group Attributes:'
keep_rows_between(input_file_path, output_file_path, start_value, end_value)

df = pd.read_csv('new_main.csv')
# OR drop rows with NaN values
df = df.dropna(subset=['Severity Id']).astype({'Severity Id': int})
coloumnlist = ["Severity Id","Issue Type Name" ,"Threat Class", "Security Risk", "Cause"]
df = df[coloumnlist].drop_duplicates().fillna('').reset_index(drop=True)

# Load the new dataset and preprocess it
new_data = df
X_new_text = new_data.drop(columns=['Severity Id'])  # Exclude the output columns if present
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
current_datetime = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file_path = f'output_data_{current_datetime}.csv'
new_data.to_csv(output_file_path, index=False)

print(f"Predictions saved to '{output_file_path}'.")
