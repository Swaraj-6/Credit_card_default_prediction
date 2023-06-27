# Credit_card_default_prediction


## Problem Statement: Credit Card Default Prediction

The problem at hand is to develop a predictive model that can accurately predict whether a credit card holder is likely to default on their payments in the near future. Predicting credit card defaults is crucial for financial institutions to assess the risk associated with extending credit and make informed decisions about lending.

## Deployment:
This project was deployed on aws using aws-beanstalk but had to take it down due to many reasons so it is deployed on render.com
[Link](https://credit-default-prediction.onrender.com)

## Dataset: 

This provided dataset contains information on default payments, demographic factors, credit data, history of payment, and bill statements of credit card clients in Taiwan from April 2005 to September 2005.


## Approach:

- #### Exploratory Data Analysis (EDA):
 Perform a thorough analysis of the dataset to gain insights into the distribution of variables, identify missing values, handle outliers, and understand the relationships between features and the target variable.

- #### Feature Engineering: 
Transform and preprocess the dataset by handling missing values, encoding categorical variables, and scaling numerical features. Create new relevant features if necessary.

- #### Model Selection: 
Select appropriate machine learning algorithms for classification, such as logistic regression, decision trees, random forests, support vector machine, knn classifier. Consider ensemble methods for improved performance.

- #### Model Training: 
Split the dataset into training and validation sets. Train the selected models on the training data and tune their hyperparameters using techniques like cross-validation or grid search.

- #### Model Evaluation: 
Evaluate the trained models on the validation set using appropriate evaluation metrics such as accuracy, precision, recall, and F1-score. Select the model with the best performance.


## IDE Used:
- Pycharm

## Language Used:
- Python

## Framework used:
- Flask
- Scikit-learn
