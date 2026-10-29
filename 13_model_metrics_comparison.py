# Online Shoppers Purchasing Intention Prediction
# Step 13: Precision, Recall and F1-Score Comparison

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load model comparison results
df = pd.read_csv("outputs/results/model_comparison.csv")

# Select evaluation metrics
metrics = ["Precision", "Recall", "F1-Score"]

# Convert results into a suitable format for plotting
plot_df = df.melt(
    id_vars="Model",
    value_vars=metrics,
    var_name="Metric",
    value_name="Score"
)

# Create comparison graph
sns.set_theme(style="whitegrid")

plt.figure(figsize=(12, 7))

ax = sns.barplot(
    data=plot_df,
    x="Model",
    y="Score",
    hue="Metric"
)

ax.set_title(
    "Model Performance Comparison: Precision, Recall and F1-Score",
    fontsize=14,
    fontweight="bold"
)

ax.set_xlabel("Machine Learning Model")
ax.set_ylabel("Score (%)")
ax.set_ylim(0, 100)

plt.xticks(rotation=20, ha="right")
plt.legend(title="Evaluation Metric")
plt.tight_layout()

# Save graph
plt.savefig(
    "outputs/plots/09_model_metrics_comparison.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()

print("Model metrics comparison graph saved successfully!")