import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
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

valueCount = df[col].value_counts()
st.subheader("Distribution of samples")
st.bar_chart(valueCount)

st.header("Model Selection")
selected_model = st.selectbox("Select a machine learning model", ["Random Forest", "Gradient Boosting", "SVM"])

if selected_model == "Random Forest":
    model = RandomForestClassifier()
elif selected_model == "Gradient Boosting":
    model = GradientBoostingClassifier()
else:
    model = SVC()

model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
st.write(f"{selected_model} Model Accuracy: {accuracy:.2f}")

st.subheader("Confusion Matrix")
conf_matrix = confusion_matrix(y_test, y_pred)
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
st.pyplot(plt)

