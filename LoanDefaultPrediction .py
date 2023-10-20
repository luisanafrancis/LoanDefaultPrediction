#!/usr/bin/env python
# coding: utf-8

# ![COUR_IPO.png](attachment:COUR_IPO.png)

# # Welcome to the Data Science Coding Challange!
# 
# Test your skills in a real-world coding challenge. Coding Challenges provide CS & DS Coding Competitions with Prizes and achievement badges!
# 
# CS & DS learners want to be challenged as a way to evaluate if they’re job ready. So, why not create fun challenges and give winners something truly valuable such as complimentary access to select Data Science courses, or the ability to receive an achievement badge on their Coursera Skills Profile - highlighting their performance to recruiters.

# ## Introduction
# 
# In this challenge, you'll get the opportunity to tackle one of the most industry-relevant machine learning problems with a unique dataset that will put your modeling skills to the test. Financial loan services are leveraged by companies across many industries, from big banks to financial institutions to government loans. One of the primary objectives of companies with financial loan services is to decrease payment defaults and ensure that individuals are paying back their loans as expected. In order to do this efficiently and systematically, many companies employ machine learning to predict which individuals are at the highest risk of defaulting on their loans, so that proper interventions can be effectively deployed to the right audience.
# 
# In this challenge, we will be tackling the loan default prediction problem on a very unique and interesting group of individuals who have taken financial loans. 
# 
# Imagine that you are a new data scientist at a major financial institution and you are tasked with building a model that can predict which individuals will default on their loan payments. We have provided a dataset that is a sample of individuals who received loans in 2021. 
# 
# This financial institution has a vested interest in understanding the likelihood of each individual to default on their loan payments so that resources can be allocated appropriately to support these borrowers. In this challenge, you will use your machine learning toolkit to do just that!

# ## Understanding the Datasets

# ### Train vs. Test
# In this competition, you’ll gain access to two datasets that are samples of past borrowers of a financial institution that contain information about the individual and the specific loan. One dataset is titled `train.csv` and the other is titled `test.csv`.
# 
# `train.csv` contains 70% of the overall sample (255,347 borrowers to be exact) and importantly, will reveal whether or not the borrower has defaulted on their loan payments (the “ground truth”).
# 
# The `test.csv` dataset contains the exact same information about the remaining segment of the overall sample (109,435 borrowers to be exact), but does not disclose the “ground truth” for each borrower. It’s your job to predict this outcome!
# 
# Using the patterns you find in the `train.csv` data, predict whether the borrowers in `test.csv` will default on their loan payments, or not.

# ### Dataset descriptions
# Both `train.csv` and `test.csv` contain one row for each unique Loan. For each Loan, a single observation (`LoanID`) is included during which the loan was active. 
# 
# In addition to this identifier column, the `train.csv` dataset also contains the target label for the task, a binary column `Default` which indicates if a borrower has defaulted on payments.
# 
# Besides that column, both datasets have an identical set of features that can be used to train your model to make predictions. Below you can see descriptions of each feature. Familiarize yourself with them so that you can harness them most effectively for this machine learning task!

# In[1]:


import pandas as pd
data_descriptions = pd.read_csv('data_descriptions.csv')
pd.set_option('display.max_colwidth', None)
data_descriptions


# ## How to Submit your Predictions to Coursera
# Submission Format:
# 
# In this notebook you should follow the steps below to explore the data, train a model using the data in `train.csv`, and then score your model using the data in `test.csv`. Your final submission should be a dataframe (call it `prediction_df` with two columns and exactly 109,435 rows (plus a header row). The first column should be `LoanID` so that we know which prediction belongs to which observation. The second column should be called `predicted_probability` and should be a numeric column representing the __likelihood that the borrower will default__.
# 
# Your submission will show an error if you have extra columns (beyond `LoanID` and `predicted_probability`) or extra rows. The order of the rows does not matter.
# 
# The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of `prediction_df` with column names `LoanID` and `predicted_probability`!
# 
# To determine your final score, we will compare your `predicted_probability` predictions to the source of truth labels for the observations in `test.csv` and calculate the [ROC AUC](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.roc_auc_score.html). We choose this metric because we not only want to be able to predict which loans will default, but also want a well-calibrated likelihood score that can be used to target interventions and support most accurately.

# ## Import Python Modules
# 
# First, import the primary modules that will be used in this project. Remember as this is an open-ended project please feel free to make use of any of your favorite libraries that you feel may be useful for this challenge. For example some of the following popular packages may be useful:
# 
# - pandas
# - numpy
# - Scipy
# - Scikit-learn
# - keras
# - maplotlib
# - seaborn
# - etc, etc

# In[2]:


# Import required packages

# Data packages
import pandas as pd
import numpy as np

# Machine Learning / Classification packages
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.dummy import DummyClassifier

# Visualization Packages
from matplotlib import pyplot as plt
import seaborn as sns
get_ipython().run_line_magic('matplotlib', 'inline')


# In[3]:


# Import any other packages you may want to use


# ## Load the Data
# 
# Let's start by loading the dataset `train.csv` into a dataframe `train_df`, and `test.csv` into a dataframe `test_df` and display the shape of the dataframes.

# In[4]:


train_df = pd.read_csv("train.csv")
print('train_df Shape:', train_df.shape)
train_df.head()


# In[5]:


test_df = pd.read_csv("test.csv")
print('test_df Shape:', test_df.shape)
test_df.head()


# ## Explore, Clean, Validate, and Visualize the Data (optional)
# 
# Feel free to explore, clean, validate, and visualize the data however you see fit for this competition to help determine or optimize your predictive model. Please note - the final autograding will only be on the accuracy of the `prediction_df` predictions.

# In[6]:


#Initial Exploration:

print('Basic Information about the train dataset')
train_df.info()


# In[7]:


#Initial Exploration:

print('Basic Information about the test dataset')
test_df.info()


# In[8]:


print('Summary statistics of train dataset')
train_df.describe()


# In[9]:


print('Summary statistics of test dataset')
test_df.describe()


# In[10]:


#Handling Missing Values:

print('Missing values of train dataset')
train_df.isnull().sum()


# In[11]:


print('Missing values of test dataset')
test_df.isnull().sum()


# In[12]:


#Checking for Outliers

plt.figure(figsize=(8, 6))
sns.boxplot(x=train_df['CreditScore'])
plt.xlabel('CreditScore')
plt.title('Box Plot of CreditScore')
plt.show()


plt.figure(figsize=(8, 6))
sns.boxplot(x=train_df['Income'])
plt.xlabel('Income')
plt.title('Box Plot of Income')
plt.show()

plt.figure(figsize=(8, 6))
sns.boxplot(x=train_df['LoanAmount'])
plt.xlabel('LoanAmount')
plt.title('Box Plot of Loan Amount')
plt.show()


# In[13]:


#Distribution of Categorical Variables

plt.figure(figsize=(8, 6))
sns.countplot(x=train_df['Education'])
plt.xlabel('Education')
plt.title('Count of Education Levels')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x=train_df['EmploymentType'])
plt.xlabel('Employment Status')
plt.title('Count of Employment Status')
plt.xticks(rotation=45)
plt.show()


plt.figure(figsize=(8, 6))
sns.countplot(x=train_df['LoanPurpose'])
plt.xlabel('Purpose of the Loan')
plt.title('Count of Purpose of the Loan')
plt.xticks(rotation=45)
plt.show()


# In[14]:


# Select the categorical columns in the DataFrame
categorical_cols = train_df.select_dtypes(include=['object']).columns

# Loop through each categorical column and apply value_counts
for col in categorical_cols:
    print(f"Value counts for column '{col}':")
    print(train_df[col].value_counts())
    print("\n")


# In[15]:


#Correlation Analysis

correlation_matrix = train_df.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()


# In[16]:


correlation_matrix = train_df.corr()
print(correlation_matrix)


# Here's how to interpret the correlation values:
# 
# - The correlation coefficient ranges from -1 to 1.
#   - A positive correlation (close to 1) indicates that as one variable increases, the other also increases.
#   - A negative correlation (close to -1) indicates that as one variable increases, the other decreases.
#   - A correlation close to 0 means little to no linear relationship between the two variables.
# 
# Here are some key observations from the correlation matrix:
# 
# 1. **Age and Default**: There is a negative correlation of approximately -0.1678 between 'Age' and 'Default'. This suggests that as the age of borrowers increases, the likelihood of default decreases (which makes sense as older individuals might be more financially stable).
# 
# 2. **Income and Default**: There is a negative correlation of approximately -0.0991 between 'Income' and 'Default'. This suggests that higher income is associated with a lower likelihood of default.
# 
# 3. **LoanAmount and Default**: There is a positive correlation of approximately 0.0867 between 'LoanAmount' and 'Default'. This indicates that larger loan amounts are associated with a higher likelihood of default.
# 
# 4. **CreditScore and Default**: There is a negative correlation of approximately -0.0342 between 'CreditScore' and 'Default'. A higher credit score is associated with a lower likelihood of default.
# 
# 5. **InterestRate and Default**: There is a positive correlation of approximately 0.1313 between 'InterestRate' and 'Default'. Higher interest rates are associated with a higher likelihood of default.
# 
# 6. **MonthsEmployed and Default**: There is a negative correlation of approximately -0.0974 between 'MonthsEmployed' and 'Default'. Longer employment periods are associated with a lower likelihood of default.
# 
# 7. **NumCreditLines, LoanTerm, and DTIRatio**: These features have relatively low correlations with 'Default', indicating weak linear relationships.
# 
# It's important to note that while these correlations provide insights into the relationships between features and the target variable, correlation does not imply causation. The relationships may be more complex, and other factors not considered in the correlation analysis may also influence loan default.
# 
# 

# ## Make predictions (required)
# 
# Remember you should create a dataframe named `prediction_df` with exactly 109,435 entries plus a header row attempting to predict the likelihood of borrowers to default on their loans in `test_df`. Your submission will throw an error if you have extra columns (beyond `LoanID` and `predicted_probaility`) or extra rows.
# 
# The file should have exactly 2 columns:
# `LoanID` (sorted in any order)
# `predicted_probability` (contains your numeric predicted probabilities between 0 and 1, e.g. from `estimator.predict_proba(X, y)[:, 1]`)
# 
# The naming convention of the dataframe and columns are critical for our autograding, so please make sure to use the exact naming conventions of `prediction_df` with column names `LoanID` and `predicted_probability`!

# ### Example prediction submission:
# 
# The code below is a very naive prediction method that simply predicts loan defaults using a Dummy Classifier. This is used as just an example showing the submission format required. Please change/alter/delete this code below and create your own improved prediction methods for generating `prediction_df`.

# **PLEASE CHANGE CODE BELOW TO IMPLEMENT YOUR OWN PREDICTIONS**

# In[17]:


### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# Fit a dummy classifier on the feature columns in train_df:
dummy_clf = DummyClassifier(strategy="stratified")
dummy_clf.fit(train_df.drop(['LoanID', 'Default'], axis=1), train_df.Default)


# In[18]:


### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# Use our dummy classifier to make predictions on test_df using `predict_proba` method:
predicted_probability = dummy_clf.predict_proba(test_df.drop(['LoanID'], axis=1))[:, 1]


# In[19]:


### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# Combine predictions with label column into a dataframe
prediction_df = pd.DataFrame({'LoanID': test_df[['LoanID']].values[:, 0],
                             'predicted_probability': predicted_probability})


# In[20]:


### PLEASE CHANGE THIS CODE TO IMPLEMENT YOUR OWN PREDICTIONS

# View our 'prediction_df' dataframe as required for submission.
# Ensure it should contain 104,480 rows and 2 columns 'CustomerID' and 'predicted_probaility'
print(prediction_df.shape)
prediction_df.head(10)


# In[21]:


#Feature engineering
import numpy as np

# Feature Crosses
train_df['Age_Income'] = train_df['Age'] * train_df['Income']

# Binning Numeric Features
bins = [0, 30, 45, 60, 100]
labels = ['Young', 'Middle-Aged', 'Senior', 'Elderly']
train_df['Age_Group'] = pd.cut(train_df['Age'], bins=bins, labels=labels)

# Logarithmic Transformation
train_df['Log_Income'] = np.log(train_df['Income'] + 1)  # Adding 1 to avoid issues with log(0)

# Feature Aggregations
education_mean_income = train_df.groupby('Education')['Income'].mean()
train_df['Education_Mean_Income'] = train_df['Education'].map(education_mean_income)

# Count Encoding
loan_purpose_count = train_df['LoanPurpose'].value_counts().to_dict()
train_df['LoanPurpose_Count'] = train_df['LoanPurpose'].map(loan_purpose_count)


# In[22]:


# Feature Crosses
test_df['Age_Income'] = test_df['Age'] * test_df['Income']

# Binning Numeric Features
bins = [0, 30, 45, 60, 100]
labels = ['Young', 'Middle-Aged', 'Senior', 'Elderly']
test_df['Age_Group'] = pd.cut(test_df['Age'], bins=bins, labels=labels)

# Logarithmic Transformation
test_df['Log_Income'] = np.log(test_df['Income'] + 1)  # Adding 1 to avoid issues with log(0)

# Feature Aggregations
education_mean_income = train_df.groupby('Education')['Income'].mean()
test_df['Education_Mean_Income'] = test_df['Education'].map(education_mean_income)

# Count Encoding
loan_purpose_count = train_df['LoanPurpose'].value_counts().to_dict()
test_df['LoanPurpose_Count'] = test_df['LoanPurpose'].map(loan_purpose_count)


# In[23]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split

# Separate the training dataset into features (X_train) and the target variable (y_train)
X_train = train_df[['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education_Mean_Income', 'LoanPurpose_Count', 'Age_Income', 'Log_Income']]
y_train = train_df['Default']

# Define the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

# Separate the test dataset into features (X_test)
X_test = test_df[['Age', 'Income', 'LoanAmount', 'CreditScore', 'MonthsEmployed', 'NumCreditLines', 'InterestRate', 'LoanTerm', 'DTIRatio', 'Education_Mean_Income', 'LoanPurpose_Count', 'Age_Income', 'Log_Income']]

# Make predictions on the test set
predicted_probabilities = model.predict_proba(X_test)[:, 1]

# Create prediction_df
prediction_df = pd.DataFrame({'LoanID': test_df['LoanID'], 'predicted_probability': predicted_probabilities})

# Ensure the DataFrame has the required number of rows (109,435)
print(prediction_df.shape)

# View the first 10 rows of prediction_df
print(prediction_df.head(10))

# Split the training dataset into training and validation sets for evaluation
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Make predictions on the validation set
y_val_pred = model.predict(X_val)

# Calculate evaluation metrics
accuracy = accuracy_score(y_val, y_val_pred)
precision = precision_score(y_val, y_val_pred)
recall = recall_score(y_val, y_val_pred)
f1 = f1_score(y_val, y_val_pred)
roc_auc = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])

# Print the evaluation metrics
print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")
print(f"ROC AUC: {roc_auc:.2f}")


# ## Final Tests - **IMPORTANT** - the cells below must be run prior to submission
# 
# Below are some tests to ensure your submission is in the correct format for autograding. The autograding process accepts a csv `prediction_submission.csv` which we will generate from our `prediction_df` below. Please run the tests below an ensure no assertion errors are thrown.

# In[24]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

# Writing to csv for autograding purposes
prediction_df.to_csv("prediction_submission.csv", index=False)
submission = pd.read_csv("prediction_submission.csv")

assert isinstance(submission, pd.DataFrame), 'You should have a dataframe named prediction_df.'


# In[25]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.columns[0] == 'LoanID', 'The first column name should be CustomerID.'
assert submission.columns[1] == 'predicted_probability', 'The second column name should be predicted_probability.'


# In[26]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.shape[0] == 109435, 'The dataframe prediction_df should have 109435 rows.'


# In[27]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

assert submission.shape[1] == 2, 'The dataframe prediction_df should have 2 columns.'


# In[28]:


# FINAL TEST CELLS - please make sure all of your code is above these test cells

## This cell calculates the auc score and is hidden. Submit Assignment to see AUC score.


# ## SUBMIT YOUR WORK!
# 
# Once we are happy with our `prediction_df` and `prediction_submission.csv` we can now submit for autograding! Submit by using the blue **Submit Assignment** at the top of your notebook. Don't worry if your initial submission isn't perfect as you have multiple submission attempts and will obtain some feedback after each submission!

# In[ ]:




