'''
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
Date: 2024-08-04
'''


import os
import shap
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import GridSearchCV
from sklearn.metrics import plot_roc_curve, classification_report


os.environ['QT_QPA_PLATFORM']='offscreen'
sns.set()


def import_data(pth):
    '''
    returns dataframe for the csv found at pth

    input:
            pth: a path to the csv
    output:
            df: pandas dataframe
    '''
    try:
        data = pd.read_csv(pth)
        return data
    except FileNotFoundError:
        print("File does not exist!")
    return None


def perform_eda(df):
    '''
    perform eda on df and save figures to images folder
    input:
            df: pandas dataframe

    output:
            None
    '''
    # General parameters
    figsize = (20, 10)
    rootpath = './images/eda'
    
    # Derived Churn variable
    df['Churn'] = df['Attribution_Flag'].apply(lambda val: 0 if val == "Existing Customer" else 1)
    
    # Figure 1 : Churn distribution
    plt.figure(figsize=figsize)
    df['Churn'].hist()
    plt.savefig(rootpath+'/churn_distribution.png')
    plt.close()
    
    # Figure 2 : Age distribution
    plt.figure(figsize=figsize)
    df['Customer_Age'].hist()
    plt.savefig(rootpath+'/age_distribution.png')
    plt.close()
    
    # Figure 3 : Marital status distribution
    plt.figure(figsize=figsize)
    df['Marital_Status'].value_counts('normalize').plot(kind='bar')
    plt.savefig(rootpath+'/marital_status_distribution.png')
    plt.close()
    
    # Figure 4 : Total transaction count distribution
    plt.figure(figsize=figsize)
    sns.histplot(df['Total_Trans_Ct'], stat='density', kde=True)
    plt.savefig(rootpath+'/total_transaction_count.png')
    plt.close()
    
    # Figure 5 : Correlations
    plt.figure(figsize=figsize)
    sns.heatmap(df.corr(), annot=False, cmap='Dark2_r', linewidths = 2)
    plt.savefig(rootpath+'/correlation_heatmap.png')
    plt.close()


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    propotion of churn for each category - associated with cell 15 from the notebook

    input:
            df: pandas dataframe
            category_lst: list of columns that contain categorical features
            response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
            df: pandas dataframe with new columns for
    '''
    for col in category_list:
        lst = []
        groups = df.groupby(col).mean()[response]
        
        for val in df[col]:
            list.append(groups.loc[val])
        
        new_encoded_name = col + "_" + response
        df[new_encoded_name] = lst
        
    return df


def perform_feature_engineering(df, response):
    '''
    input:
              df: pandas dataframe
              response: string of response name [optional argument that could be used for naming variables or index y column]

    output:
              X_train: X training data
              X_test: X testing data
              y_train: y training data
              y_test: y testing data
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category'
    ]
    
    keep_cols = [
        'Customer_Age',
        'Dependent_count', 
        'Months_on_book',
        'Total_Relationship_Count', 
        'Months_Inactive_12_mon',
        'Contacts_Count_12_mon', 
        'Credit_Limit', 
        'Total_Revolving_Bal',
        'Avg_Open_To_Buy', 
        'Total_Amt_Chng_Q4_Q1', 
        'Total_Trans_Amt',
        'Total_Trans_Ct', 
        'Total_Ct_Chng_Q4_Q1', 
        'Avg_Utilization_Ratio',
        'Gender_Churn',
        'Education_Level_Churn',
        'Marital_Status_Churn',
        'Income_Category_Churn',
        'Card_Category_Churn'
    ]
    
    # Target variable
    y = df['Churn']
    
    # Encode categorical variables by each ratios
    df = encoder_helper(df, cat_columns, response)
    X = df[keep_cols]
    
    # Train/Test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    return X_train, X_test, y_train, y_test

    
def classification_report_image(y_train,
                                y_test,
                                y_train_preds_lr,
                                y_train_preds_rf,
                                y_test_preds_lr,
                                y_test_preds_rf):
    '''
    produces classification report for training and testing results and stores report as image
    in images folder
    input:
            y_train: training response values
            y_test:  test response values
            y_train_preds_lr: training predictions from logistic regression
            y_train_preds_rf: training predictions from random forest
            y_test_preds_lr: test predictions from logistic regression
            y_test_preds_rf: test predictions from random forest

    output:
             None
    '''
    
    # Random forest model : Classification result
    plt.figure(figsize=(15,10))
    plt.text(0.01, 1.25, str('Random Forest Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_test, y_test_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Random Forest Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_train, y_train_preds_rf)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/rf_classification_report.png')

    
    # Logistic regression model : Classification result
    plt.figure(figsize=(15,10))
    plt.text(0.01, 1.25, str('Logistic Regression Train'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.05, str(classification_report(y_train, y_train_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.6, str('Logistic Regression Test'), {'fontsize': 10}, fontproperties = 'monospace')
    plt.text(0.01, 0.7, str(classification_report(y_test, y_test_preds_lr)), {'fontsize': 10}, fontproperties = 'monospace')
    plt.axis('off')
    plt.savefig('./images/results/logistic_classification_report.png')

    

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
    # Calculate feature importances
    importances = model.feature_importances_
    
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]
    
    # Rearrange feature names so they match the sorted feature importance
    names = [X_data.columns[i] for i in indices]
    
    # Create plot
    plt.figure(figsize=(20, 5))
    
    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel("Importance")
    
    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)
    
    # Save plot
    plt.savefig(output_path)
    

    
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
    # grid search
    rfc = RandomForestClassifier(random_state=42)
    lrc = LogisticRegressions(solver='lbfgs', max_iter=3000)
    
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    
    lrc.fit(X_train, y_train)
    
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)
    
    # Save the best model
    joblib.dump(cv_rfc.best_estimator_, './models/rfc_model.pkl')
    joblib.dump(lrc, './models/logistic_model.pkl')

    # Save the classification report plots
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_train_preds_rf,
        y_test_preds_lr,
        y_test_preds_rf)                             
    
    # Save plot image
    lrc_plot = plot_roc_curve(lrc, X_test, y_test)
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    rfc_disp = plot_roc_curve(cv_rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    lrc_plot.plot(ax=ax, alpha=0.8)
    plt.savefig('./images/results/roc_curve_result.png')
    
    # Save feature importance plot
    feature_importance_plot(cv_rfc.best_estimator_, X_train, './images/results/feature_importances.png')
    
    
if __name__ == "__main__":
    
    raw_data = import_data("./data/bank_data.csv")
    perform_eda(raw_data)
    X_train, X_test, y_train, y_test = perform_feature_engineering(raw_data, 'Churn')
    train_models(X_tarin, X_test, y_train, y_test)
