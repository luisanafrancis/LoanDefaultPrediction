# Loan Default Prediction - Data Science Coding Challenge

This project involves building a machine learning model to predict loan defaults using logistic regression. The primary goal is to develop a model that accurately predicts whether a borrower will default on a loan and provides well-calibrated likelihood scores for decision-making and interventions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Getting Started](#getting-started)
3. [Requirements](#requirements)
4. [Data Exploration](#data-exploration)
5. [Feature Engineering](#feature-engineering)
6. [Model Development](#model-development)
7. [Model Evaluation](#model-evaluation)
8. [Results](#results)
9. [Contributors](#contributors)

## Project Overview

In this project, we build a logistic regression model to predict loan defaults. The model is trained on a labeled dataset containing various features related to borrowers and their loan applications. The project includes data exploration, feature engineering, model development, and model evaluation.

## Getting Started

These instructions will help you set up and run the project on your local machine.

### Requirements

- Python (3.6+)
- Jupyter Notebook or a Python development environment
- Required Python libraries (scikit-learn, pandas, numpy)

### Data Exploration

- The project begins with data exploration to understand the dataset's structure and characteristics.
- Statistical summaries and visualizations help identify trends and patterns in the data.

### Feature Engineering

- Feature engineering is a critical step where we transform and create new features from the raw data.
- Techniques used include feature crosses, binning, logarithmic transformation, feature aggregations, and count encoding.

### Model Development

- We employ logistic regression as the primary modeling technique for predicting loan defaults.
- The model is trained using the labeled training dataset.

### Model Evaluation

- The model's performance is evaluated on a validation set, and various metrics (accuracy, precision, recall, F1 score) are computed.

### Results

- The project concludes with a submission of predicted probabilities for loan defaults on a test dataset.
- The ROC AUC score obtained after submission was 0.7443.
- The ROC AUC metric reflects the model's ability to distinguish between loan defaults and non-defaults.

## Contributors

- Luisana Francis

