# Credit Card Fraud Detection Using Machine Learning

## Overview
This project explores the use of machine learning techniques to detect fraudulent credit card transactions. Fraud detection is a major challenge for financial institutions because fraudulent transactions represent a very small percentage of the overall transaction volume.

## Business Problem
Financial institutions process millions of transactions daily, making manual fraud detection impossible. The objective of this project is to develop machine learning models capable of identifying fraudulent transactions while minimizing false positives.

## Dataset
The project uses the Credit Card Fraud Detection dataset from Kaggle.  
The dataset contains 284,807 transactions with 492 fraudulent cases.

## Methods
The following steps were performed:

- Exploratory Data Analysis (EDA)
- Handling class imbalance using SMOTE
- Model training using:
  - Logistic Regression
  - Random Forest
  - XGBoost
- Model evaluation using:
  - Precision
  - Recall
  - F1-score
  - Precision–Recall Curve

## Key Findings
The dataset is highly imbalanced because fraudulent transactions represent a very small fraction of the total data. Precision–recall analysis helps evaluate the trade-off between detecting more fraud cases and avoiding false positives.

## Ethical Considerations
Fraud detection systems must protect customer privacy and avoid incorrectly flagging legitimate transactions as fraudulent.

## Tools
Python  
pandas  
scikit-learn  
matplotlib  
seaborn  

## Files
- Project report: `680project3-final-milestone3.pdf`
