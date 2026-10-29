# Online Shoppers Purchasing Intention Prediction
# Step 12: Model Performance Comparison

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Create output folders
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("outputs/plots", exist_ok=True)

# Load training and testing data
X_train = pd.read_csv("outputs/results/X_train.csv")
X_test = pd.read_csv("outputs/results/X_test.csv")
y_train = pd.read_csv("outputs/results/y_train.csv").values.ravel()
y_test = pd.read_csv("outputs/results/y_test.csv").values.ravel()

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "KNN": KNeighborsClassifier(n_neighbors=5),
    "Decision Tree": DecisionTreeClassifier(max_depth=10, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "Naive Bayes": GaussianNB(),
    "SVM": SVC()
}

# Scale data for KNN and SVM
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Store model performance
results = []

# Train and evaluate models
for name, model in models.items():
    if name in ["KNN", "SVM"]:
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, y_pred) * 100,
        "Precision": precision_score(y_test, y_pred, zero_division=0) * 100,
        "Recall": recall_score(y_test, y_pred, zero_division=0) * 100,
        "F1-Score": f1_score(y_test, y_pred, zero_division=0) * 100
    })

# Create comparison table
results_df = pd.DataFrame(results)

metric_columns = ["Accuracy", "Precision", "Recall", "F1-Score"]
results_df[metric_columns] = results_df[metric_columns].round(2)

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

# Save comparison results
results_df.to_csv("outputs/results/model_comparison.csv", index=False)

# Plot model accuracy comparison
sns.set_theme(style="whitegrid")
plt.figure(figsize=(11, 6))

ax = sns.barplot(
    data=results_df,
    x="Model",
    y="Accuracy"
)

# Display accuracy values
for bar in ax.patches:
    value = bar.get_height()
    ax.annotate(
        f"{value:.2f}%",
        (bar.get_x() + bar.get_width() / 2, value),
        ha="center",
        va="bottom",
        fontsize=9,
        xytext=(0, 4),
        textcoords="offset points"
    )

ax.set_title(
    "Machine Learning Model Accuracy Comparison",
    fontsize=15,
    fontweight="bold"
)
ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(0, 105)

plt.xticks(rotation=20, ha="right")
plt.tight_layout()

plt.savefig(
    "outputs/plots/07_model_accuracy_comparison.png",
    dpi=300,
    bbox_inches="tight"
)
plt.show()

# Identify model with the highest F1-score
best_model = results_df.loc[results_df["F1-Score"].idxmax()]

print("\nBest Model Based on F1-Score:", best_model["Model"])
print("F1-Score:", best_model["F1-Score"], "%")

# BEST MODEL CONFUSION MATRIX


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Identify the model with the highest F1-score
best_model_name = results_df.loc[
    results_df["F1-Score"].idxmax(), "Model"
]

# Train the best model again
best_model = models[best_model_name]

if best_model_name in ["KNN", "SVM"]:
    best_model.fit(X_train_scaled, y_train)
    best_predictions = best_model.predict(X_test_scaled)
else:
    best_model.fit(X_train, y_train)
    best_predictions = best_model.predict(X_test)

# Generate confusion matrix
cm = confusion_matrix(
    y_test,
    best_predictions,
    labels=[0, 1]
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=["No Purchase", "Purchase"]
)

fig, ax = plt.subplots(figsize=(8, 6))

display.plot(
    ax=ax,
    cmap="Blues",
    values_format="d",
    colorbar=True
)

ax.set_title(
    f"Confusion Matrix - {best_model_name}",
    fontsize=14,
    fontweight="bold"
)

plt.tight_layout()

plt.savefig(
    "outputs/plots/08_best_model_confusion_matrix.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Best model confusion matrix saved successfully!")
print("\nComparison completed successfully!")