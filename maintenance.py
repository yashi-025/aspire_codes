import os
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

# Load datasets
os.chdir(r'D:\Desktop\vscode\projects\python\apna college\Predictive Maintenance')
train_data = pd.read_csv('train_FD001.txt', sep='\s+', header=None)
test_data = pd.read_csv('test_FD001.txt', sep='\s+', header=None)
rul_data = pd.read_csv('RUL_FD001.txt', sep='\s+', header=None)

# Add column names
column_names = ['engine_id', 'cycle', 'setting1', 'setting2', 'setting3'] + [f'sensor{i}' for i in range(1, 22)]
train_data.columns = column_names
test_data.columns = column_names
rul_data.columns = ['RUL']

# Calculate RUL for training data
rul_train = train_data.groupby('engine_id')['cycle'].max().reset_index()
rul_train.columns = ['engine_id', 'max_cycle']
train_data = train_data.merge(rul_train, on='engine_id')
train_data['RUL'] = train_data['max_cycle'] - train_data['cycle']
train_data.drop('max_cycle', axis=1, inplace=True)

# Prepare test data
test_rul = test_data.groupby('engine_id')['cycle'].max().reset_index()
test_rul.columns = ['engine_id', 'max_cycle']
test_data = test_data.merge(test_rul, on='engine_id')
test_data = test_data.merge(rul_data, left_on='engine_id', right_index=True)
test_data['RUL'] = test_data['RUL'] + test_data['max_cycle'] - test_data['cycle']
test_data.drop('max_cycle', axis=1, inplace=True)

# Separate features and target
X_train = train_data.drop(['RUL', 'engine_id'], axis=1)
y_train = train_data['RUL']
X_test = test_data.drop(['RUL', 'engine_id'], axis=1)
y_test = test_data['RUL']

# Initialize the model
model = RandomForestRegressor(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate RMSE
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
print(f'Root Mean Squared Error: {rmse}')

# Save predictions to a text file
predictions = pd.DataFrame({'engine_id': test_data['engine_id'], 'RUL': y_pred})
predictions.to_csv('predicted_RUL.txt', index=False, header=True, sep='\t')

# Print to check the saved file
print(predictions.head())
