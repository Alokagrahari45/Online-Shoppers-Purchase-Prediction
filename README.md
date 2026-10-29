# 🛒 Online Shoppers Purchasing Intention Prediction

### 🚀 Machine Learning & Data Science Project

## 📌 1. Project Overview
This project analyzes online shopping sessions and predicts whether a visitor will make a purchase using machine learning classification algorithms.

## 🎯 2. Objectives
- 🔍 Inspect and understand the dataset.
- 🧹 Remove duplicate records.
- 📊 Perform Exploratory Data Analysis (EDA).
- ⚙️ Preprocess data for machine learning.
- 🤖 Implement six classification algorithms.
- 📈 Compare model performance using evaluation metrics.
- 🏆 Identify the best-performing model using F1-Score.

## 📂 3. Dataset Information
- **Original Records:** 12,330
- **Original Features:** 18
- **Target Variable:** `Revenue`
- **Problem Type:** Binary Classification
- `True` = Purchase
- `False` = No Purchase
- **Cleaned Records:** 12,205 after duplicate removal.

## 🛠️ 4. Technologies Used
- 🐍 Python
- 🐼 Pandas
- 📊 Matplotlib
- 🎨 Seaborn
- 🤖 Scikit-learn
- 💻 Visual Studio Code

## 🔄 5. Project Workflow
1. Data Inspection
2. Data Cleaning
3. Exploratory Data Analysis
4. Data Preprocessing
5. Train-Test Splitting (80:20)
6. Model Training
7. Model Evaluation
8. Performance Comparison

## 📊 6. Exploratory Data Analysis
The project includes four visualizations:
- 📊 Purchase vs Non-Purchase Sessions — Bar Graph
- 📈 Monthly Visitor Sessions Trend — Line Graph with Value Labels
- 📊 Monthly Purchase Conversion Rate — Bar Graph with Percentage Labels
- 📦 PageValues by Purchase Outcome — Box Plot

📁 Graphs are saved in `outputs/plots/`.

## 🤖 7. Machine Learning Models
| No. | Algorithm | Implementation |
|---:|---|---|
| 1 | Logistic Regression | `06_logistic_regression.py` |
| 2 | K-Nearest Neighbors (KNN) | `07_knn_model.py` |
| 3 | Decision Tree | `08_decision_tree.py` |
| 4 | Random Forest | `09_random_forest.py` |
| 5 | Gaussian Naive Bayes | `10_naive_bayes.py` |
| 6 | Support Vector Machine (SVM) | `11_svm_model.py` |

## 📏 8. Evaluation Metrics
- 🎯 Accuracy
- 🔍 Precision
- 📈 Recall
- ⚖️ F1-Score
- 🧮 Confusion Matrix

These metrics help evaluate model performance, including the identification of purchase sessions in an imbalanced dataset.

## 🏆 9. Model Comparison
All six models are evaluated using the same training and testing split. Their Accuracy, Precision, Recall, and F1-Score are compared. The model with the highest F1-Score is selected as the best-performing model.

📁 Results: `outputs/results/model_comparison.csv`

📈 Model Accuracy Graph: `outputs/plots/07_model_accuracy_comparison.png`

🧮 Best Model Confusion Matrix: `outputs/plots/08_best_model_confusion_matrix.png`

📊 Metrics Comparison Graph: `outputs/plots/09_model_metrics_comparison.png`

## 📁 10. Project Structure
```text
Online_Shoppers_Purchase_Prediction/
├── .venv/
├── data/
│   ├── shopping.csv
│   ├── cleaned_shopping.csv
│   └── preprocessed_shopping.csv
├── models/
│   ├── 06_logistic_regression.py
│   ├── 07_knn_model.py
│   ├── 08_decision_tree.py
│   ├── 09_random_forest.py
│   ├── 10_naive_bayes.py
│   └── 11_svm_model.py
├── outputs/
│   ├── plots/
│   └── results/
├── 00_README.md
├── 01_data_preparation.py
├── 12_model_comparison.py
└── 13_model_metrics_comparison.py
```

## ▶️ 11. How to Run
Run commands from the project root directory in the VS Code terminal.

**Step 1 — Data Preparation**
```bash
python 01_data_preparation.py
```

**Step 2 — Train and Compare All Models**
```bash
python 12_model_comparison.py
```

**Step 3 — Generate Metrics Comparison Graph**
```bash
python 13_model_metrics_comparison.py
```

Individual models can also be run using commands such as:
```bash
python models/06_logistic_regression.py
python models/07_knn_model.py
python models/08_decision_tree.py
python models/09_random_forest.py
python models/10_naive_bayes.py
python models/11_svm_model.py
```

## 💡 12. Learning Outcomes
- Data Inspection, Cleaning, and Preprocessing
- Exploratory Data Analysis and Visualization
- Categorical Feature Encoding
- Supervised Machine Learning
- Classification Evaluation Metrics
- Model Performance Comparison

## 🔮 13. Future Scope
- Hyperparameter Tuning
- Class-Imbalance Handling
- ROC-AUC Analysis
- Web-Based Purchase Prediction Interface

## ✅ Conclusion
This project applies data analysis and six machine learning classification algorithms to understand online shopping behavior and predict purchase intention.

**Domain:** Machine Learning & Data Science  
**Language:** Python  
**Problem Type:** Binary Classification