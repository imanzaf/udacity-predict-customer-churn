'''
This file tests all functions in churn_library.py

Author: Iman Zafar
Date: April 2023
'''

# import libraries
import os
import logging
import time
import pytest
import pandas as pd
import churn_library as cl

# set up logging file
logging.basicConfig(
    filename="./logs/churn_library_{}.log".format(time.strftime('%b_%d_%Y_%H_%M_%S')),
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s',
    force=True)


# create fixture for path
@pytest.fixture(scope='module')
def path():
    return "./data/bank_data.csv"


def test_import(path):
    '''
    test data import
    '''
    # check if file present
    try:
        df = cl.import_data(path)
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    # check if file populated
    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err

    # update variable in Namespace
    pytest.df = df


def test_create_churn_col():
    '''
    test create churn (response) column function
    '''
    # check if Attrition_Flag present
    try:
        df = pytest.df
        df = cl.create_churn_col(df)
        logging.info("Testing create_churn_col: SUCCESS")
    except NameError as err:
        logging.error(
            "Testing create_churn_col: The dataframe doesn't have the required Attrition_Flag column")
        raise err

    # check if Attrition_Flag populated
    try:
        assert df['Churn'].isna().sum() != len(df['Churn'])
    except AssertionError as err:
        logging.error("Testing create_churn_col: Churn column is null")
        raise err

    # update variable in Namespace
    pytest.df = df


def test_cat_eda_plot():
    '''
    test categorical eda plot function
    '''
    # define arguments for function
    df = pytest.df
    feature = 'Education_Level'
    output_path = './images/eda/univariate_cat'

    # check if plot file generated
    try:
        cl.cat_eda_plot(df, feature, output_path)
        assert os.path.exists('{}/{}_barplot.png'.format(output_path, feature))
        logging.info('Testing cat_eda_plot: SUCCESS')
    except AssertionError as err:
        logging.error('Testing cat_eda_plot: The image file was not created')
        raise err


def test_num_eda_plot():
    '''
    test numerical eda plot function
    '''
    # define arguments for function
    df = pytest.df
    feature = 'Customer_Age'
    output_path = './images/eda/univariate_num'

    # check if plot file generated
    try:
        cl.num_eda_plot(df, feature, output_path)
        assert os.path.exists('{}/{}_hist.png'.format(output_path, feature))
        logging.info('Testing num_eda_plot: SUCCESS')
    except AssertionError as err:
        logging.error('Testing num_eda_plot: The image file was not created')
        raise err


def test_multivariate_eda_plot():
    '''
    test multivariate eda plot function
    '''
    # define arguments for function
    df = pytest.df
    output_path = './images/eda/multivariate'

    # check if correlation plot file generated
    try:
        cl.multivariate_eda_plot(df, output_path)
        assert os.path.exists('{}/df_heatmap.png'.format(output_path))
        logging.info('Testing multivariate_eda_plot (heatmap): SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing multivariate_eda_plot (heatmap) : The image file was not created')
        raise err

    # define feature args for function
    feat_1 = 'Credit_Limit'
    feat_2 = 'Months_on_book'

    # check if bivariate plot file generated
    try:
        cl.multivariate_eda_plot(df, output_path, feat_1, feat_2)
        assert os.path.exists(
            '{}/{}_vs_{}.png'.format(output_path, feat_1, feat_2))
        logging.info('Testing multivariate_eda_plot (bivariate plot): SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing multivariate_eda_plot (bivariate plot): The image file was not created')
        raise err


# create fixture for response variable
@pytest.fixture(scope='module')
def response():
    return 'Churn'


# create fixture for categorical cols list
@pytest.fixture(scope='module')
def cat_cols():
    cat_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
    return cat_cols


# create fixture for numeric cols list
@pytest.fixture(scope='module')
def quant_cols():
    quant_cols = [
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
        'Avg_Utilization_Ratio']
    return quant_cols


def test_encoder_helper(cat_cols, response):
    '''
    test encoder helper function

    :param cat_cols: list of categorical columns for input in encoder_helper function
    :param response: (str) name of response column for input in encoder_helper function
    '''
    # define df for function
    df = pytest.df

    # check if encoded columns created
    try:
        df, encoded_cols = cl.encoder_helper(df, cat_cols, response)
        assert df[encoded_cols].shape[0] > 0
        assert df[encoded_cols].shape[1] > 0
        logging.info('Testing encoder_helper: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing encoder_helper: encoded columns were not created')
        raise err


def test_perform_feature_engineering(quant_cols, cat_cols, response):
    '''
    test feature engineering function

    :param quant_cols: list of numeric columns for input in perform_feature_engineering()
    :param cat_cols: list of categorical columns for input in perform_feature_engineering()
    :param response: (str) name of response column for input in perform_feature_engineering()
    '''
    # define df for function
    df = pytest.df

    # check if engineered data is of correct shape
    try:
        X_train, X_test, y_train, y_test = cl.perform_feature_engineering(
            df, quant_cols, cat_cols, response)

        assert X_train.shape[0] > 0
        assert X_test.shape[0] > 0
        assert X_train.shape[1] == len(quant_cols + cat_cols)
        assert X_test.shape[1] == len(quant_cols + cat_cols)

        assert y_train.shape[0] > 1
        assert y_test.shape[0] > 1
        assert len(y_train.shape) == 1
        assert len(y_test.shape) == 1

        logging.info('Testing perform_feature_engineering: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing perform_feature_engineering: rows and columns not populated as expected')
        raise err

    # update variables in Namespace
    pytest.X_train = X_train
    pytest.X_test = X_test
    pytest.y_train = y_train
    pytest.y_test = y_test


def test_train_logistic_reg():
    '''
    test train logistic regression function
    '''
    # define args for function
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train

    # check if model and predictions created
    try:
        lrc, y_train_preds_lr, y_test_preds_lr = cl.train_logistic_reg(
            X_train, X_test, y_train)
        assert lrc is not None
        assert y_train_preds_lr.shape[0] > 0
        assert y_test_preds_lr.shape[0] > 0
        logging.info('Testing train_logistic_reg: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing train_logistic_reg: Logistic Regression model and predictions were not generated')
        raise err

    # save variables in Namespace
    pytest.lrc = lrc
    pytest.train_preds_lrc = y_train_preds_lr
    pytest.test_preds_lrc = y_test_preds_lr


def test_train_random_forest():
    '''
    test train random forest function
    '''
    # define args for function
    X_train = pytest.X_train
    X_test = pytest.X_test
    y_train = pytest.y_train

    # check if model and predictions created
    try:
        cv_rfc, y_train_preds_rf, y_test_preds_rf = cl.train_random_forest(
            X_train, X_test, y_train)
        assert cv_rfc is not None
        assert y_train_preds_rf.shape[0] > 0
        assert y_test_preds_rf.shape[0] > 0
        logging.info('Testing train_random_forest: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing train_random_forest: Random Forest model and predictions were not generated')
        raise err

    # update variables in Namespace
    pytest.rf = cv_rfc
    pytest.train_preds_rf = y_train_preds_rf
    pytest.test_preds_rf = y_test_preds_rf


def test_save_model():
    '''
    test save model function
    '''
    # define args for function
    model = pytest.lrc
    model_name = 'Logistic Regression'
    output_path = './models'

    try:
        cl.save_model(model, model_name, output_path)
        assert os.path.exists(
            '{}/{}.pkl'.format(output_path, model_name.lower().replace(' ', '_')))
        logging.info('Testing save_model: SUCCESS')
    except AssertionError as err:
        logging.error('Testing save_model: model not saved')
        raise err


def test_classification_report_image():
    '''
    test classification report function
    '''
    # define args for function
    y_train = pytest.y_train
    y_test = pytest.y_test
    y_train_preds = pytest.train_preds_lrc
    y_test_preds = pytest.test_preds_lrc
    model_name = 'Logistic Regression'
    output_path = './images/results'

    # check if classification report image created
    try:
        cl.classification_report_image(
            y_train,
            y_test,
            y_train_preds,
            y_test_preds,
            model_name,
            output_path)
        assert os.path.exists(
            '{}/{}_classification_report.png'.format(
                output_path, model_name.lower().replace(
                    ' ', '_')))
        logging.info('Testing classification_report_image: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing classification_report_image: classification report not saved')
        raise err


def test_roc_curve_image():
    '''
    test roc curve function
    '''
    # define args for function
    rfc = pytest.rf
    lrc = pytest.lrc
    X_test = pytest.X_test
    y_test = pytest.y_test
    output_path = './images/results'

    # check if ROC plot image created
    try:
        cl.roc_curve_image(rfc, lrc, X_test, y_test, output_path)
        assert os.path.exists('{}/roc_curve.png'.format(output_path))
        logging.info('Testing roc_curve_image: SUCCESS')
    except AssertionError as err:
        logging.error('Testing roc_curve_image: roc curve image not saved')
        raise err


def test_feature_importance_plot():
    '''
    test feature importance function
    '''
    # define args for function
    model = pytest.rf
    X_test = pytest.X_test
    X_train = pytest.X_train
    X_data = pd.concat((X_train, X_test), axis=0)
    output_path = './images/results'

    # check if feature importance plot image created
    try:
        cl.feature_importance_plot(model.best_estimator_, X_data, output_path)
        assert os.path.exists('{}/feature_importances.png'.format(output_path))
        logging.info('Testing feature_importance_plot: SUCCESS')
    except AssertionError as err:
        logging.error(
            'Testing feature_importance_plot: feature importance plot not saved')
        raise err
