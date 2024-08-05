# Predict Customer Churn Using Production-Level Code

In this project, the goal is to convert a provided research notebook into a development/production ready code, applying Clean Code Principles.
This is part of the ML DevOps Engineer Nanodegree Udacity.

<br>

## Project Description
This project uses credit card customers dataset and we aim to identify who are most likely to churn. The completed project includes a Python package that follows coding and engineering best practices for implementing software like:
- Error handling
- Refactoring : Modular, Documented
- PEP8 conventions : checked with 'pylint' and 'autopep8'
- Logging

<br>

## Data description
Original Customer Churn Dataset contains 28 columns. Since the main purpose of this project is not to elaborate the modeling, the provided research notebook used simple EDA and modeling methods.
- Consists of categorical and numerical columns (e.g. Age, Gender, Education_Level, Income_Cateogory..)
- Target variable to predict is 'Attribution_Flag', which is a binary classification.
- Models used : Logistic Regression and Random Forests with Grid Search

<br>

## Files description
This project contains:
- Guide.ipynb: the project guideline provided by Udacity
- churn_notebook.ipynb: provided data analysis and modeling process which contains:
* Load dataset
* Exploratory Data Analysis (EDA) and pre-processing
* Feature Engineering
* Training and testing
* Classification Report generation

<br>

Provdided 'churn_notebook.ipynb' was refactored for a production-ready code in two files:
* churn_library.py: contains five refactored functions
* churn_script_logging_and_tests.py: testing functions in 'churn_library.py' and logging

<br>

## Folder structure
- 'requirements_py3.10.txt' : installation versions
- 'data/'': folder which stores the original data
- 'images/'': folder for EDA and classification report images
- 'models/'': folder for generated models in pkl extension
- 'logs/'': folder for logging

<br>

## Running Files
1. Create virtual environment
2. Install python dependencies
3. Run the refactored data analysis and modeling module
4. Run the test & logging module

'''bash
conda create -n my_venv python=3.10.0
conda activate my_venv
conda install pip
pip install -r requirements_py3.10.txt
python churn_library.py
python churn_script_logging_and_tests.py
'''

<br>

If you want to check the PEP8 score of these files:
'''bash
pylint churn_library.py
pylint churn_script_logging_and_tests.py
'''

You can also use 'autopep8' to automatically edit and improve the PEP8 score:
'''bash
autopep8 --in-place --aggressive --aggressive churn_library.py
'''

<br>
