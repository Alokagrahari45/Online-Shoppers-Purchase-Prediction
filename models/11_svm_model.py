# Online Shoppers Purchasing Intention Prediction

# Practical: Support Vector Machine (SVM)

import pandas as pd
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Load training and testing data

X_train = pd.read_csv("outputs/results/X_train.csv")
X_test = pd.read_csv("outputs/results/X_test.csv")
y_train = pd.read_csv("outputs/results/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/results/y_test.csv").values.ravel()

# Scale the data

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create SVM model

model = SVC()

# Train model

model.fit(X_train, y_train)

# Make predictions

y_pred = model.predict(X_test)

# Check model performance

accuracy = accuracy_score(y_test, y_pred)

print("SVM Model")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
