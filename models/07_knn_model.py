import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler


# Load training and testing data
X_train = pd.read_csv("outputs/results/X_train.csv")
X_test = pd.read_csv("outputs/results/X_test.csv")
y_train = pd.read_csv("outputs/results/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/results/y_test.csv").values.ravel()

# Scale the data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create and train KNN model
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Check model performance
accuracy = accuracy_score(y_test, y_pred)

print("KNN Model")
print("Accuracy:", round(accuracy * 100, 2), "%")
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("Classification Report:")
print(classification_report(y_test, y_pred))
