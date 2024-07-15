import os
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier

# Download NLTK data
nltk.download('stopwords')

# Load training and testing datasets
os.chdir(r'D:\Desktop\vscode\projects\python\apna college\disaster')
train_data = pd.read_csv('train.csv')  # Replace with your training dataset path
test_data = pd.read_csv('test.csv')    # Replace with your testing dataset path

# Check the actual column names in your datasets
print(train_data.columns)
print(test_data.columns)

# Preprocess the text data
def preprocess_text(text):
    text = re.sub(r'http\S+', '', str(text))  # Remove URLs
    text = re.sub(r'[^A-Za-z\s]', '', str(text))  # Remove non-alphanumeric characters
    text = text.lower()  # Convert to lowercase
    text = text.split()
    stop_words = set(stopwords.words('english'))
    text = [word for word in text if word not in stop_words]
    text = ' '.join(text)
    return text

# Apply preprocessing to both training and testing datasets
train_data['clean_text'] = train_data['tweet'].apply(preprocess_text)  # Replace 'tweet' with your actual tweet column name
test_data['clean_text'] = test_data['tweet'].apply(preprocess_text)    # Replace 'tweet' with your actual tweet column name

# Split data into features and target
X_train = train_data['clean_text']
y_train = train_data['disaster']  # Replace 'disaster' with your actual target column name

X_test = test_data['clean_text']

# Initialize TF-IDF Vectorizer
vectorizer = TfidfVectorizer(max_features=5000)

# Fit and transform the training data
X_train_tfidf = vectorizer.fit_transform(X_train)

# Transform the test data
X_test_tfidf = vectorizer.transform(X_test)

# Initialize the model
rf_model = RandomForestClassifier(random_state=42)

# Train the model
rf_model.fit(X_train_tfidf, y_train)

# Predict on the test set
y_pred = rf_model.predict(X_test_tfidf)

# Add predictions to the test data
test_data['disaster'] = y_pred

# Save the predictions to a new CSV file
test_data.to_csv('test_with_predictions.csv', index=False)

print("Predictions have been saved to 'test_with_predictions.csv'.")