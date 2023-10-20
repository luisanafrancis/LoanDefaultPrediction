# Loan Default Prediction - Data Science Coding Challenge

## Overview

This project is a Data Science Coding Challenge by Coursera, focusing on loan default prediction. We were provided with two datasets: `train_df` and `test_df`. The primary goal was to build a predictive model to identify individuals at risk of loan default. After exploring various machine learning models, including Random Forest and Decision Tree, we found that Logistic Regression outperformed the others.

## Datasets

- **Training Dataset (`train_df`):** Contains information about individuals who received loans in 2021. Features include age, income, loan amount, credit score, and more. The 'Default' column serves as the target variable (0 for non-default, 1 for default).

- **Testing Dataset (`test_df`):** Similar to the training dataset but without the 'Default' column. This dataset was used to evaluate the model's performance and make predictions.

## Model Selection

We initially experimented with Random Forest and Decision Tree models. However, Logistic Regression emerged as the model of choice due to its simplicity and superior performance. The model demonstrates a balance between ease of interpretation and effective prediction of loan default risk.

## Approach

1. **Data Preprocessing:**
   - Handled categorical features through one-hot encoding.
   - Split the dataset into training and testing sets for model evaluation.
   - Addressed class imbalance issues if present.

2. **Performance Evaluation:**
   - Assessed the model's performance using key metrics such as accuracy, precision, recall, F1 score, and ROC AUC.
   - Logistic Regression displayed strong performance, particularly in accuracy and ROC AUC.

3. **Next Steps:**
   - Ongoing monitoring and fine-tuning of the model's performance.
   - Exploration of feature engineering, hyperparameter tuning, or alternative models if necessary.
   - Potential implementation of the model in real-world lending processes to inform decisions and reduce potential losses.

---

This summary provides a succinct overview of the project's objectives, data, model selection, and approach, focusing on the choice of Logistic Regression as the preferred model for loan default prediction.
