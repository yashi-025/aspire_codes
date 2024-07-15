import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Set working directory (change to your working directory if necessary)
os.chdir(r'D:\Desktop\vscode\projects\python\apna college\rainfall prediction')

# Load the weather data
weather_data = pd.read_csv('weatherAUS.csv')

# Load the txt files
train_fd = pd.read_csv('train_FD001.txt', sep='\s+', header=None)
test_fd = pd.read_csv('test_FD001.txt', sep='\s+', header=None)
rul_fd = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None)

# Display the first few rows of each file to understand their structure
print(weather_data.head())
print(train_fd.head())
print(test_fd.head())
print(rul_fd.head())

# Assuming train_fd and test_fd have similar structure and columns
# Assuming they can be concatenated to form additional features
# Preprocess the txt files (here we simply concatenate for simplicity)
combined_fd = pd.concat([train_fd, test_fd], axis=0)

# Assuming the txt files have the same number of rows as the weather data
# and can be concatenated column-wise (this is just an assumption)
combined_fd.columns = [f'feature_{i}' for i in range(combined_fd.shape[1])]

# Concatenate the features from the txt files with the weather data
combined_data = pd.concat([weather_data.reset_index(drop=True), combined_fd.reset_index(drop=True)], axis=1)

# Fill missing values with mean or mode (depending on the column type)
for column in combined_data.columns:
    if combined_data[column].dtype == 'object':
        combined_data[column] = combined_data[column].fillna(combined_data[column].mode()[0])
    else:
        combined_data[column] = combined_data[column].fillna(combined_data[column].mean())

# Convert categorical features to numerical using one-hot encoding
combined_data = pd.get_dummies(combined_data)

# Ensure 'RainTomorrow' exists or define it based on RISK_MM
if 'RainTomorrow' not in combined_data.columns:
    if 'RISK_MM' in combined_data.columns:
        combined_data['RainTomorrow'] = combined_data['RISK_MM'].apply(lambda x: 'Yes' if x > 0 else 'No')
    else:
        raise ValueError("Column 'RainTomorrow' or 'RISK_MM' not found in the dataset.")

# Split data into features and target
X = combined_data.drop('RainTomorrow', axis=1)
y = combined_data['RainTomorrow']

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize the Random Forest Classifier
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train_scaled, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test_scaled)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, pos_label='Yes')
recall = recall_score(y_test, y_pred, pos_label='Yes')
f1 = f1_score(y_test, y_pred, pos_label='Yes')

# Print the results
print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-score: {f1}")

# Write the results to a text file
output_file_path = os.path.join(os.getcwd(), 'rainfall_prediction_results.txt')

try:
    with open(output_file_path, 'w') as file:
        file.write(f"Accuracy: {accuracy}\n")
        file.write(f"Precision: {precision}\n")
        file.write(f"Recall: {recall}\n")
        file.write(f"F1-score: {f1}\n")
    print(f"Results successfully written to {output_file_path}")
except IOError as e:
    print(f"An error occurred while writing to the file: {e}")



# import os
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.preprocessing import StandardScaler
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # Set working directory (change to your working directory)
# os.chdir(r'D:\Desktop\vscode\projects\python\apna college\rainfall prediction')

# # Load the weather data
# weather_data = pd.read_csv('weatherAUS.csv')

# # Load the txt files
# train_fd = pd.read_csv('train_FD001.txt', sep='\s+', header=None)
# test_fd = pd.read_csv('test_FD001.txt', sep='\s+', header=None)
# rul_fd = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None)

# # Display the first few rows of each file to understand their structure
# print(weather_data.head())
# print(train_fd.head())
# print(test_fd.head())
# print(rul_fd.head())

# # Assuming train_fd and test_fd have similar structure and columns
# # Assuming they can be concatenated to form additional features
# # Preprocess the txt files (here we simply concatenate for simplicity)
# combined_fd = pd.concat([train_fd, test_fd], axis=0)

# # Assuming the txt files have the same number of rows as the weather data
# # and can be concatenated column-wise (this is just an assumption)
# combined_fd.columns = [f'feature_{i}' for i in range(combined_fd.shape[1])]

# # Concatenate the features from the txt files with the weather data
# combined_data = pd.concat([weather_data.reset_index(drop=True), combined_fd.reset_index(drop=True)], axis=1)

# # Fill missing values with mean or mode (depending on the column type)
# for column in combined_data.columns:
#     if combined_data[column].dtype == 'object':
#         combined_data[column] = combined_data[column].fillna(combined_data[column].mode()[0])
#     else:
#         combined_data[column] = combined_data[column].fillna(combined_data[column].mean())

# # Convert categorical features to numerical using one-hot encoding
# combined_data = pd.get_dummies(combined_data)

# # Ensure 'RainTomorrow' exists or define it based on RISK_MM
# if 'RainTomorrow' not in combined_data.columns:
#     if 'RISK_MM' in combined_data.columns:
#         combined_data['RainTomorrow'] = combined_data['RISK_MM'].apply(lambda x: 'Yes' if x > 0 else 'No')
#     else:
#         raise ValueError("Column 'RainTomorrow' or 'RISK_MM' not found in the dataset.")

# # Split data into features and target
# X = combined_data.drop('RainTomorrow', axis=1)
# y = combined_data['RainTomorrow']

# # Split into training and test sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Scale the features
# scaler = StandardScaler()
# X_train_scaled = scaler.fit_transform(X_train)
# X_test_scaled = scaler.transform(X_test)

# # Initialize the Random Forest Classifier
# rf_model = RandomForestClassifier(random_state=42)

# # Train the model
# rf_model.fit(X_train_scaled, y_train)

# # Predict on the test set
# y_pred = rf_model.predict(X_test_scaled)

# # Evaluate the model
# accuracy = accuracy_score(y_test, y_pred)
# precision = precision_score(y_test, y_pred, pos_label='Yes')
# recall = recall_score(y_test, y_pred, pos_label='Yes')
# f1 = f1_score(y_test, y_pred, pos_label='Yes')

# print(f"Accuracy: {accuracy}")
# print(f"Precision: {precision}")
# print(f"Recall: {recall}")
# print(f"F1-score: {f1}")
