'''
This module performs tests each function in churn_library.py

- test data import
- test EDA
- test encoder helper
- test feature engineering
- test train models

PEP8 conventions checked:

>> pylint churn_script_logging_and_tests.py # 8.46/10
>> autopep8 churn_script_logging_and_tests.py

Folder structure:

- models produced are stored in './models'
- EDA plots and classification reports are stored in './images'
- Logs are stored in './logs/'

Author: Seulgie Han
Date: 2024-08-05
'''

import os
import logging
import pytest
import churn_library as cls

logging.basicConfig(
    filename='./logs/churn_library_test.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


def import_raw_data():
    '''
    import raw data for later test cases
    '''
    data = cls.import_data("./data/bank_data.csv")
    if data is None or data.empty:
        raise ValueError("import_data returned no data")
    return data


def test_import(raw_data):
    '''
    Test data import
    '''
    try:
        assert raw_data.shape[0] > 0 # No rows in the imported data
        assert raw_data.shape[1] > 0 # No columns in the imported data 
        logging.info("Testing import_data: SUCCESS")
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


def test_eda(raw_data):
    '''
    Test perform_eda function
    '''
    try:
        cls.perform_eda(raw_data)
        logging.info("Testing perform_eda: SUCCESS")
    except Exception as err:
        logging.error("Testing perform_eda: ERROR")
        raise err


def test_encoder_helper(raw_data):
    '''
    Test encoder_helper
    '''
    cat_columns = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']

    try:
        data = cls.encoder_helper(raw_data, cat_columns, "Churn")
        for col in cat_columns:
            assert col + "_Churn" in data.columns
        logging.info("TESTING encoder_helper: SUCCESS")
    except AssertionError as err:
        logging.error(
            "TESTING encoder_helper: ERROR - Some categorical columns are missing")
        raise err


def test_perform_feature_engineering(raw_data):
    '''
    Test perform_feature_engineering
    '''
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        raw_data, "Churn")

    try:
        assert X_train.shape[0] > 0 # X_train has no rows
        assert X_test.shape[0] > 0 # X_test has no rows
        assert y_train.shape[0] > 0 # y_train has no rows
        assert y_test.shape[0] > 0 # y_test has no rows
        logging.info("TESTING perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error(
            "TESTING perform_feature_engineering: ERROR - Unexpected size")
        raise err


def test_train_models(raw_data):
    '''
    Test train_models
    '''
    X_train, X_test, y_train, y_test = cls.perform_feature_engineering(
        raw_data, "Churn")

    # Perform model training
    cls.train_models(X_train, X_test, y_train, y_test)

    # Check if the results images are saved
    try:
        result_images = os.listdir("./images/results/")
        assert len(result_images) > 0
        logging.info("TESTING saving results images: SUCCESS - Results images saved")
    except AssertionError as err:
        logging.error(
            "TESTING train_models: ERROR - Some result images are missing")
        raise err

    # Check if the models are saved
    try:
        models = os.listdir("./models/")
        assert len(models) > 0
        logging.info("TESTING train_models saved: SUCCESS - Models saved")
    except AssertionError as err:
        logging.error("TESTING train_models saved: ERROR - Some models are missing")
        raise err


if __name__ == "__main__":
    # if we change the file name starting with 'test_', then 
    # we can write : pytest.main() and run in the command line as
    # pytest test_churn_library.py -v
    # Source: https://docs.pytest.org/en/stable/
    # Manually running tests instead of using pytest fixture
    try:
        raw_data = import_raw_data()
        test_import(raw_data)
        test_eda(raw_data)
        test_encoder_helper(raw_data)
        test_perform_feature_engineering(raw_data)
        test_train_models(raw_data)
        print("All tests ran successfully.")
    except Exception as e:
        print(f"An error occured: {e}")
