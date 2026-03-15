
# Credit Card Fraud Detection Project
# Author: Fatma Mahfoud

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, precision_recall_curve, roc_curve
from sklearn.metrics import auc
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv("creditcard.csv")

print("Dataset shape:", df.shape)
print(df.head())

# Figure 1
plt.figure(figsize=(6,4))
sns.countplot(x="Class", data=df)
plt.title("Distribution of Fraudulent vs Legitimate Transactions")
plt.savefig("figure1_distribution.png")
plt.close()

# Figure 2
plt.figure(figsize=(6,4))
sns.histplot(df["Amount"], bins=50)
plt.title("Transaction Amount Distribution")
plt.savefig("figure2_amount.png")
plt.close()

# Figure 3
plt.figure(figsize=(6,4))
sns.histplot(df["Time"], bins=50)
plt.title("Time-Based Transaction Patterns")
plt.savefig("figure3_time.png")
plt.close()

# Heatmap
plt.figure(figsize=(10,8))
corr = df.corr()
sns.heatmap(corr, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.savefig("figure4_heatmap.png")
plt.close()

# Prepare data
X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# Train model
model = LogisticRegression(max_iter=1000)
model.fit(X_train_res, y_train_res)

y_scores = model.predict_proba(X_test)[:,1]
y_pred = model.predict(X_test)

print(classification_report(y_test, y_pred))

# Precision Recall Curve
precision, recall, _ = precision_recall_curve(y_test, y_scores)
pr_auc = auc(recall, precision)

plt.figure(figsize=(6,4))
plt.plot(recall, precision)
plt.title("Precision–Recall Curve for Fraud Detection Model")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.savefig("figure5_prcurve.png")
plt.close()

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_scores)
roc_auc = auc(fpr, tpr)

plt.figure(figsize=(6,4))
plt.plot(fpr, tpr)
plt.title("ROC Curve for Fraud Detection Model")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.savefig("figure6_roc.png")
plt.close()

print("Project finished. Figures saved in current folder.")
