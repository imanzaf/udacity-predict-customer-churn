'''
This is a library of functions to find customers who are likely to churn.
The functions complete Data Science processes including:
    EDA, Feature Engineering, Model Training, Prediction, and Model Evaluation

Author: Iman Zafar
Date: April 2023
'''

# import libraries
from sklearn.metrics import plot_roc_curve, classification_report
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def import_data(pth):
    '''
    Returns dataframe for the csv found at pth

    :param pth: a path to the csv

    :returns df: pandas dataframe
    '''
    df = pd.read_csv(pth)
    return df


def create_churn_col(df):
    '''
    Create binary column for whether a customer has attrited or not

    :param df: pandas dataframe with 'Attrition_Flag' column

    :return df: pandas dataframe with 'Churn' column
    '''
    df['Churn'] = df['Attrition_Flag'].apply(
        lambda val: 0 if val == "Existing Customer" else 1)
    return df


def cat_eda_plot(df, feature, output_path, style='bmh'):
    '''
    Perform eda on categorical feature of df - plot barplot of value counts and save figure as image

    :param df: pandas dataframe
    :param feature: (str) categorical feature name
    :param output_path: (str) path to save image to
    :param style: (str - optional) matplotlib style sheet name

    :returns: None
    '''
    plt.style.use(style)
    plt.figure(figsize=(20, 10))
    plt.title('{} Bar Plot'.format(feature))
    df[feature].value_counts('normalize').plot(kind='bar')
    plt.savefig('{}/{}_barplot.png'.format(output_path,
                feature), bbox_inches='tight')
    plt.close()


def num_eda_plot(df, feature, output_path, style='bmh'):
    '''
    Perform eda on numerical feature of df - plot histogram with kde and save figure as image

    :param df: pandas dataframe
    :param feature: (str) numerical feature name
    :param output_path: (str) path to save image to
    :param style: (str - optional) matplotlib style sheet name

    :returns: None
    '''
    plt.style.use(style)
    plt.figure(figsize=(20, 10))
    plt.title('{} Histogram with KDE'.format(feature))
    sns.histplot(df[feature], stat='density', kde=True)
    plt.savefig('{}/{}_hist.png'.format(output_path,
                feature), bbox_inches='tight')
    plt.close()


def multivariate_eda_plot(
        df,
        output_path,
        feature1=None,
        feature2=None,
        style='bmh'):
    '''
    Perform eda on two or more numerical features of df - plot correlation plot of both features
        If no feature names specified, heatmap of all numerical features created
        Figure saved as image

    :param df: pandas dataframe
    :param feature1: (str - optional) numerical feature name
    :param feature2: (str - optional) numerical feature name
    :param output_path: (str) path to save image to
    :param style: (str - optional) matplotlib style sheet name

    :returns: None
    '''
    plt.style.use(style)
    if (feature1 is None) and (feature2 is None):
        plt.figure(figsize=(20, 10))
        plt.title('Correlation Heatmap')
        sns.heatmap(df.corr(), annot=False, linewidths=2)
        plt.savefig(
            '{}/df_heatmap.png'.format(output_path),
            bbox_inches='tight')
        plt.close()
    elif (feature1 is not None) and (feature2 is not None):
        plt.figure(figsize=(20, 10))
        plt.title('{} vs {} Scatter Plot'.format(feature1, feature2))
        plt.scatter(feature1, feature2, data=df)
        plt.savefig('{}/{}_vs_{}.png'.format(output_path,
                    feature1, feature2), bbox_inches='tight')
        plt.close()
    else:
        print('Both features need to be specified for bivariate analysis')


def encoder_helper(df, category_lst, response):
    '''
    helper function to turn each categorical column into a new column with
    proportion of churn for each category - associated with cell 15 from the notebook

    :param df: pandas dataframe
    :param category_lst: list of columns that contain categorical features
    :param response: string of response name

    :returns df: pandas dataframe with new columns for columns in category_lst
    :returns encoded_cols: list of names of encoded columns
    '''
    encoded_cols = []
    for feat in category_lst:
        groups = df.groupby(feat).mean()[response]
        df['{}_{}'.format(feat, response)] = df[feat].apply(
            lambda x: groups.loc[x])
        encoded_cols.append('{}_{}'.format(feat, response))
    return df, encoded_cols


def perform_feature_engineering(df, numeric_lst, category_lst, response):
    '''
    Perform encoding of categorical features using encoder_helper() function
        and split data into test and training

    :param df: pandas dataframe
    :param numeric_lst: list of columns that contain numeric features
    :param category_lst: list of columns that contain categorical features
    :param response: string of response name

    :returns X_train: X training data
    :returns X_test: X testing data
    :returns y_train: y training data
    :returns y_test: y testing data
    '''
    df, encoded_cols = encoder_helper(df, category_lst, response)
    X, y = df[numeric_lst + encoded_cols], df[response]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test


def train_logistic_reg(X_train, X_test, y_train):
    '''
    Train Logistic Regression model, return model and model predictions, and save model

    :param X_train: X training data
    :param X_test: X testing data
    :param y_train: y training data

    :returns y_train_preds_lr: training predictions from logistic regression
    :returns y_test_preds_lr: test predictions from logistic regression
    :returns lrc: logistic regression model object
    '''
    # train Logistic Regression model
    lrc = LogisticRegression(solver='lbfgs', max_iter=3000)
    lrc.fit(X_train, y_train)
    y_train_preds_lr = lrc.predict(X_train)
    y_test_preds_lr = lrc.predict(X_test)

    return lrc, y_train_preds_lr, y_test_preds_lr


def train_random_forest(X_train, X_test, y_train):
    '''
    Train Random Forest Classifier model with cross validation, return model and model predictions

    :param X_train: X training data
    :param X_test: X testing data
    :param y_train: y training data

    :returns y_train_preds_rf: training predictions from random forest classifier
    :returns y_test_preds_rf: test predictions from random forest classifier
    :returns cv_rfc.best_estimator_: random forest classifier model object containing best estimator
    '''
    # train Random Forest Classifier model
    rfc = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [200, 500],
        'max_features': ['auto', 'sqrt'],
        'max_depth': [4, 5, 100],
        'criterion': ['gini', 'entropy']
    }
    cv_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv=5)
    cv_rfc.fit(X_train, y_train)
    y_train_preds_rf = cv_rfc.best_estimator_.predict(X_train)
    y_test_preds_rf = cv_rfc.best_estimator_.predict(X_test)

    return cv_rfc, y_train_preds_rf, y_test_preds_rf


def save_model(model, model_name, output_path):
    '''
    Save model as pkl file

    :param model: model object
    :param model_name: (str) model name
    :param output_path: (str) path to save model to

    :return: None
    '''
    joblib.dump(model, '{}/{}.pkl'.format(output_path,
                model_name.lower().replace(' ', '_')))


def classification_report_image(y_train, y_test,
                                y_train_preds, y_test_preds,
                                model_name, output_path):
    '''
    Produce classification report for training and testing results and store report as image

    :param y_train: training response values
    :param y_test:  test response values
    :param y_train_preds: training predictions from model
    :param y_test_preds: test predictions from model
    :param model_name: (str) model name for title
    :param output_path: (str) path to save report to

    :returns: None
    '''
    plt.rc('figure', figsize=(5, 5))
    plt.text(0.01, 1.25, str('{} Train'.format(model_name)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.05, str(
            classification_report(
                y_test, y_test_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.text(0.01, 0.6, str('{} Test'.format(model_name)),
             {'fontsize': 10}, fontproperties='monospace')
    plt.text(
        0.01, 0.7, str(
            classification_report(
                y_train, y_train_preds)), {
            'fontsize': 10}, fontproperties='monospace')
    plt.axis('off')
    plt.savefig('{}/{}_classification_report.png'.format(output_path,
                model_name.lower().replace(' ', '_')), bbox_inches='tight')
    plt.close()


def roc_curve_image(rfc, lrc, X_test, y_test, output_path):
    '''
    Produce ROC curve plot for Logistic Regression
        and Random Forest Classifier models and store plot as image

    :param rfc: Random Forest model object containing best_estimator_
    :param lrc: Logistic Regression trained model object
    :param X_test: X test data
    :param y_test: response test data
    :param output_path: (str) path to save report to

    :returns: None
    '''
    plt.figure(figsize=(15, 8))
    ax = plt.gca()
    plot_roc_curve(rfc.best_estimator_, X_test, y_test, ax=ax, alpha=0.8)
    plot_roc_curve(lrc, X_test, y_test, ax=ax, alpha=0.8)
    plt.savefig('{}/roc_curve.png'.format(output_path), bbox_inches='tight')
    plt.close()


def feature_importance_plot(model, X_data, output_path):
    '''
    Creates and stores the feature importance plot

    :param model: model object containing feature_importances_
    :param X_data: pandas dataframe of X values
    :param output_path: path to store the figure

    :returns: None
    '''
    # Calculate feature importances
    importances = model.feature_importances_
    # Sort feature importances in descending order
    indices = np.argsort(importances)[::-1]

    # Rearrange feature names so they match the sorted feature importances
    names = [X_data.columns[i] for i in indices]

    # Create plot
    plt.figure(figsize=(20, 5))

    # Create plot title
    plt.title("Feature Importance")
    plt.ylabel('Importance')

    # Add bars
    plt.bar(range(X_data.shape[1]), importances[indices])
    # Add feature names as x-axis labels
    plt.xticks(range(X_data.shape[1]), names, rotation=90)

    plt.savefig(
        '{}/feature_importances.png'.format(output_path),
        bbox_inches='tight')
    plt.close()


def main():
    '''
    Run library functions to complete DS processes:
        EDA, Feature Engineering, Model Training, Prediction, and Model Evaluation
    on bank_data.csv data
    '''

    # import data
    df = import_data('./data/bank_data.csv')
    df = create_churn_col(df)

    # lists of categorical and numeric columns
    cat_cols = [
        'Gender',
        'Education_Level',
        'Marital_Status',
        'Income_Category',
        'Card_Category']
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

    # EDA - create and save plots
    for col in cat_cols:
        cat_eda_plot(df, col, './images/eda/univariate_cat')
    for col in quant_cols:
        num_eda_plot(df, col, './images/eda/univariate_num')
    multivariate_eda_plot(
        df,
        './images/eda/multivariate',
        feature1='Customer_Age',
        feature2='Total_Trans_Amt')
    multivariate_eda_plot(
        df,
        './images/eda/multivariate',
        feature1='Credit_Limit',
        feature2='Months_on_book')
    multivariate_eda_plot(df, './images/eda/multivariate')

    # Feature Engineering
    X_train, X_test, y_train, y_test = perform_feature_engineering(
        df, quant_cols, cat_cols, 'Churn')

    # Train and Save Models
    lrc, y_train_preds_lr, y_test_preds_lr = train_logistic_reg(
        X_train, X_test, y_train, './models')
    cv_rfc, y_train_preds_rf, y_test_preds_rf = train_random_forest(
        X_train, X_test, y_train, './models')
    # Save Models
    save_model(lrc, 'Logistic Regression', './models')
    save_model(cv_rfc, 'Random Forest', './models')

    # Classification Report
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_lr,
        y_test_preds_lr,
        'Logistic Regression',
        './images/results')
    classification_report_image(
        y_train,
        y_test,
        y_train_preds_rf,
        y_test_preds_rf,
        'Random Forest',
        './images/results')

    # ROC Curve
    roc_curve_image(cv_rfc, lrc, X_test, y_test, './images/results')

    # Feature Importance
    feature_importance_plot(cv_rfc.best_estimator_, pd.concat(
        (X_train, X_test), axis=0), output_path='./images/results')


if __name__ == '__main__':
    main()
