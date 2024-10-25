import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler

# Load the dataset.
df = pd.read_csv('heart_disease_cleaned.csv')

# Preview the data.
print(df.head())

# Convert categorical variables to numerical.
df = pd.get_dummies(df, drop_first=True)  # Converts categorical columns to dummy variables.

# Define features and target variable.
X = df.drop(columns=['id', 'num'])  # Drop 'id' and the target column 'num'.
y = df['num']  # Target variable.

# Scale features.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data into training and testing sets.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train Random Forest Classifier.
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions.
y_pred = model.predict(X_test)

# Evaluate the model.
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))