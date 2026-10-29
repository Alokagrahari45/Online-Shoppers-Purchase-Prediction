# Online Shoppers Purchasing Intention Prediction
# Data Inspection, Cleaning, EDA, Preprocessing and Splitting

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split



# CREATE OUTPUT FOLDERS


os.makedirs("outputs/plots", exist_ok=True)
os.makedirs("outputs/results", exist_ok=True)
os.makedirs("data", exist_ok=True)



# STEP 1: DATA INSPECTION


df = pd.read_csv("data/shopping.csv")

print("\n========== DATA INSPECTION ==========")

print("Dataset Shape:", df.shape)

print("\nFirst Five Rows:")
print(df.head())

print("\nData Types:")
print(df.dtypes)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nDuplicate Rows:", df.duplicated().sum())

print("\nStatistical Summary:")
print(df.describe())

print("\nRevenue Distribution:")
print(df["Revenue"].value_counts())


# STEP 2: DATA CLEANING


print("\n========== DATA CLEANING ==========")

df = df.drop_duplicates().copy()

print("Shape After Cleaning:", df.shape)
print("Duplicates Remaining:", df.duplicated().sum())

df.to_csv("data/cleaned_shopping.csv", index=False)

print("Cleaned dataset saved successfully!")



# STEP 3: EXPLORATORY DATA ANALYSIS (EDA)


print("\n========== EXPLORATORY DATA ANALYSIS ==========")

sns.set_theme(style="whitegrid", context="notebook")

plt.rcParams["figure.dpi"] = 120
plt.rcParams["savefig.dpi"] = 300

no_purchase_color = "#457B9D"
purchase_color = "#2A9D8F"

month_order = [
    "Feb", "Mar", "May", "June", "Jul",
    "Aug", "Sep", "Oct", "Nov", "Dec"
]



# GRAPH 1: PURCHASE VS NON-PURCHASE (BAR GRAPH)


fig, ax = plt.subplots(figsize=(9, 6))

sns.countplot(
    data=df,
    x="Revenue",
    hue="Revenue",
    order=[False, True],
    palette={
        False: no_purchase_color,
        True: purchase_color
    },
    legend=False,
    ax=ax
)

total = len(df)

for bar in ax.patches:
    count = int(bar.get_height())
    percentage = count / total * 100

    ax.annotate(
        f"{count:,}\n({percentage:.1f}%)",
        (bar.get_x() + bar.get_width() / 2, bar.get_height()),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        xytext=(0, 5),
        textcoords="offset points"
    )

ax.set_title(
    "Purchase vs Non-Purchase Sessions",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Purchase Outcome")
ax.set_ylabel("Number of Sessions")
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Purchase", "Purchase"])
ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    "outputs/plots/01_revenue_distribution.png",
    bbox_inches="tight"
)

plt.show()



# GRAPH 2: MONTHLY VISITOR SESSIONS (LINE GRAPH)


monthly_sessions = (
    df.groupby("Month")
    .size()
    .reindex(month_order)
    .dropna()
)

fig, ax = plt.subplots(figsize=(11, 6))

# Use numeric positions so all labels align correctly
x_positions = range(len(monthly_sessions))

ax.plot(
    x_positions,
    monthly_sessions.values,
    marker="o",
    markersize=9,
    linewidth=2.5,
    color="#2878B5"
)

# Label every point with its session count
for x, sessions in zip(
    x_positions,
    monthly_sessions.values
):
    ax.annotate(
        f"{int(sessions):,}",
        (x, sessions),
        textcoords="offset points",
        xytext=(0, 10),
        ha="center",
        fontsize=9,
        fontweight="bold"
    )

ax.set_xticks(list(x_positions))
ax.set_xticklabels(monthly_sessions.index)

ax.set_title(
    "Monthly Visitor Sessions Trend",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Number of Sessions", fontsize=12)
ax.grid(True, linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

ax.margins(y=0.15)

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "outputs/plots/02_monthly_sessions_trend.png",
    bbox_inches="tight"
)

plt.show()



# GRAPH 3: MONTHLY PURCHASE RATE (BAR GRAPH)


monthly_purchase = (
    df.groupby("Month")["Revenue"]
    .mean()
    .mul(100)
    .reindex(month_order)
    .dropna()
)

fig, ax = plt.subplots(figsize=(11, 6))

sns.barplot(
    x=monthly_purchase.index,
    y=monthly_purchase.values,
    hue=monthly_purchase.index,
    palette="viridis",
    legend=False,
    ax=ax
)

# Label each bar with its purchase rate
for bar in ax.patches:
    rate = bar.get_height()

    ax.annotate(
        f"{rate:.1f}%",
        (
            bar.get_x() + bar.get_width() / 2,
            rate
        ),
        ha="center",
        va="bottom",
        fontsize=10,
        fontweight="bold",
        xytext=(0, 6),
        textcoords="offset points"
    )

ax.set_title(
    "Monthly Purchase Conversion Rate",
    fontsize=16,
    fontweight="bold",
    pad=18
)

ax.set_xlabel("Month", fontsize=12)
ax.set_ylabel("Purchase Rate (%)", fontsize=12)

ax.set_ylim(
    0,
    max(monthly_purchase.max() * 1.2, 1)
)

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(
    "outputs/plots/03_monthly_purchase_rate.png",
    bbox_inches="tight"
)

plt.show()



#pagevalues by purchase outocome(Box Plot)


fig, ax = plt.subplots(figsize=(9, 6))

sns.boxplot(
    data=df,
    x="Revenue",
    y="PageValues",
    hue="Revenue",
    order=[False, True],
    palette={
        False: no_purchase_color,
        True: purchase_color
    },
    legend=False,
    ax=ax
)

ax.set_title(
    "PageValues by Purchase Outcome",
    fontsize=15,
    fontweight="bold",
    pad=15
)

ax.set_xlabel("Purchase Outcome")
ax.set_ylabel("PageValues")
ax.set_xticks([0, 1])
ax.set_xticklabels(["No Purchase", "Purchase"])

ax.grid(axis="y", linestyle="--", alpha=0.35)
ax.set_axisbelow(True)

plt.tight_layout()

plt.savefig(
    "outputs/plots/04_page_values_boxplot.png",
    bbox_inches="tight"
)

plt.show()


print("\nAll 4 EDA graphs saved successfully!")

#data preprocessing


print("\n DATA PREPROCESSING ")

df = pd.get_dummies(
    df,
    columns=["Month", "VisitorType"],
    dtype=int
)

df["Weekend"] = df["Weekend"].astype(int)
df["Revenue"] = df["Revenue"].astype(int)

df.to_csv(
    "data/preprocessed_shopping.csv",
    index=False
)

print("Preprocessed dataset saved successfully!")
print("Total Rows:", len(df))
print("Total Columns:", len(df.columns))


#train test split

print("\nTRAIN-TEST SPLITTING")

X = df.drop("Revenue", axis=1)
y = df["Revenue"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("Training Data:", X_train.shape)
print("Testing Data:", X_test.shape)

X_train.to_csv(
    "outputs/results/X_train.csv",
    index=False
)

X_test.to_csv(
    "outputs/results/X_test.csv",
    index=False
)

y_train.to_csv(
    "outputs/results/y_train.csv",
    index=False
)

y_test.to_csv(
    "outputs/results/y_test.csv",
    index=False
)

print("Training and testing data saved successfully!")

print("\nDATA PREPARATION COMPLETED!")