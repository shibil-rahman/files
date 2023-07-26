import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib
from datetime import datetime


# Load the trained model and preprocessor from ensemble_model.pkl
joblib.load('ensemble_model.pkl')

clf = model_data['model']
preprocessor = model_data['preprocessor']

# Read the input CSV file without the 'Result' column
input_file_path = 'sast-deployment_test_input.csv'
df_input = pd.read_csv(input_file_path)

# Preprocess the input data using the loaded preprocessor
X_input = preprocessor.transform(df_input)

# Use the loaded model to make predictions on the preprocessed data
predicted_labels = clf.predict(X_input)

# Add the predicted labels as a new column 'Result' to the input DataFrame
df_input['Result'] = predicted_labels

# Save the DataFrame with the new 'Result' column to a new CSV file
current_date = datetime.now().strftime("%Y%m%d")
output_file_path = f'output_data_{current_date}.csv'
df_input.to_csv(output_file_path, index=False)

print(f"Predictions saved to '{output_file_path}'.")
