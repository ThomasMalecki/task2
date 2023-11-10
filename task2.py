import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load data
data = 'breast-cancer.data'
df = pd.read_csv(data, sep=r',', skiprows=1, names=["Class", "Age", "Menopause", "Tumor_Size",
                                                     "Inv_Nodes", "Node_Caps", "Deg_Malig",
                                                     "Breast", "Breast_Quad", "Irradiat"])

# Replace '?' with NaN
df.replace('?', np.nan, inplace=True)

# Fill NaN values with the mode
df = df.fillna(df.mode().iloc[0])

# One-hot encode categorical columns
df = pd.get_dummies(df, columns=["Age", "Menopause", "Tumor_Size",
                                  "Inv_Nodes", "Node_Caps",
                                  "Breast", "Breast_Quad", "Irradiat"])

# Split data into features (X) and target (y)
col = df.columns[0]
X = df.drop(columns=[col])
y = df[col]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Random Forest Model
random_forest_model = RandomForestClassifier()
random_forest_model.fit(X_train, y_train)
random_forest_predictions = random_forest_model.predict(X_test)
random_forest_accuracy = accuracy_score(y_test, random_forest_predictions) * 100

# Support Vector Machine Model
svm_model = SVC(kernel='linear')
svm_model.fit(X_train, y_train)
svm_predictions = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, svm_predictions) * 100

# Gradient Boosting Model
gradient_boosting_model = GradientBoostingClassifier()
gradient_boosting_model.fit(X_train, y_train)
gradient_boosting_predictions = gradient_boosting_model.predict(X_test)
gradient_boosting_accuracy = accuracy_score(y_test, gradient_boosting_predictions) * 100

# Plot the distribution of the target variable
plt.figure(figsize=(8, 6))
df[col].value_counts().plot(kind='bar', color='skyblue')
plt.xlabel(col)
plt.ylabel('Number of Observations')
plt.title(f'Distribution of {col}')
st.pyplot()

# Display accuracy scores
st.write(f"Random Forest Accuracy: {random_forest_accuracy:.2f}%")
st.write(f"SVM Accuracy: {svm_accuracy:.2f}%")
st.write(f"Gradient Boosting Accuracy: {gradient_boosting_accuracy:.2f}%")
