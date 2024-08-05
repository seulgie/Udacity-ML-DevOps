"""
This module will perform necessary data science processing steps:
- load the dataset
- perform Exploratory Data Analysis (EDA)
- perform Feature Engineering
- model Train and test
- generate reports

This script follows Clean Code Principles:
- Modularized code
- PEP8
- Error handling
- Logging and test : churn_script_logging_and_test.py

Author: Seulgie Han
Date: 2024-08-05
"""

import os
import logging
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM'] = 'offscreen'
sns.set()

def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    logging.info("Data imported successfully from %s", pth)
    return df

        
def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe
    output:
            None
    '''
    figsize = (20, 10)
    rootpath = './images/eda'
    os.makedirs(rootpath, exist_ok=True)

    df['Churn'] = df['Attrition_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    plt.figure(figsize=figsize)
    df['Churn'].hist()
    plt.savefig(os.path.join(rootpath, 'churn_distribution.png'))
    plt.close()

    plt.figure(figsize=figsize)
    df['Customer_Age'].hist()
    plt.savefig(os.path.join(rootpath, 'age_distribution.png'))
    plt.close()

    plt.figure(figsize=figsize)
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(os.path.join(rootpath, 'marital_status_distribution.png'))
    plt.close()

    plt.figure(figsize=figsize)
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(os.path.join(rootpath, 'total_transaction_count.png'))
    plt.close()

    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths=2)
    plt.savefig(os.path.join(rootpath, 'correlation_heatmap.png'))
    plt.close()

    
def encoder_helper(df, category_lst, response='Churn'):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name
    output:
            df: pandas dataframe with new columns for
    '''
    
    # Automatically detect categorical columns
    # Source: https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.select_dtypes.html
    category_lst = list(df.select_dtypes(['object']).columns)
    
    # Create Derived variables of churn ratios by each category
    for col in category_lst:
        col_groups = df.groupby(col).mean()[response]
        derived_col = col + '_' + response
        df[derived_col] = df[col].apply(lambda x: col_groups.loc[x])

    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender', 'Education_Level', 'Marital_Status', 
        'Income_Category', 'Card_Category'
    ]
    
    keep_cols = [
        'Customer_Age', 'Dependent_count', 'Months_on_book', 
        'Total_Relationship_Count', 'Months_Inactive_12_mon', 
        'Contacts_Count_12_mon', 'Credit_Limit', 'Total_Revolving_Bal', 
        'Avg_Open_To_Buy', 'Total_Amt_Chng_Q4_Q1', 'Total_Trans_Amt', 
        'Total_Trans_Ct', 'Total_Ct_Chng_Q4_Q1', 'Avg_Utilization_Ratio',
        'Gender_Churn', 'Education_Level_Churn', 'Marital_Status_Churn',
        'Income_Category_Churn', 'Card_Category_Churn'
    ]
    
    df = encoder_helper(df, cat_columns, response)
    X = df[keep_cols]
    y = df['Churn']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    logging.info("Feature engineering performed and data split into train and test sets")
    return X_train, X_test, y_train, y_test


def classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder

    input:
            y_train: training response values
            y_test: test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest
    output:
             None
    '''
    rootpath = './images/results'
    os.makedirs(rootpath, exist_ok=True)

    plt.figure(figsize=(15, 10))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(rootpath, 'rf_classification_report.png'))

    plt.figure(figsize=(15, 10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig(os.path.join(rootpath, 'logistic_classification_report.png'))

    
def feature_importance_plot(model, X_data, output_pth):
    '''
    creates and stores the feature importances in pth

    input:
            model: model object containing feature_importances_
            X_data: pandas dataframe of X values
            output_pth: path to store the figure
    output:
             None
    '''
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    names = [X_data.columns[i] for i in indices]

    plt.figure(figsize=(20, 5))
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    plt.bar(range(X_data.shape[1]), importances[indices])
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    plt.savefig(output_pth)
    

def train_models(X_train, X_test, y_train, y_test):
    '''
    train, store model results: images + scores, and store models

    input:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    output:
              None
    '''
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)

    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    lrc.fit(X_train, y_train)

    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    classification_report_image(y_train, y_test, y_train_preds_lr, y_train_preds_rf, y_test_preds_lr, y_test_preds_rf)

    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')

    feature_importance_plot(cv_rfc.best_estimator_, X_train, './images/results/feature_importances.png')


if __name__ == "__main__":
    raw_data = import_data("./data/bank_data.csv")
    perform_eda(raw_data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(raw_data, 'Churn')
    train_models(X_train, X_test, y_train, y_test)
