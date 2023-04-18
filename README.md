# Udacity - Predict Customer Churn

This repository stores the first project completed towards the Machine Learning DevOps Engineer Nanodegree from Udacity

<br/>

### Project Description
The focus of this project is to implement software engineering and coding best practices to find credit card customers 
that are most likely to churn, following the steps below. 
1. Refactor provided code and write functions to complete Data Science processes, including: EDA, 
Feature Engineering, Model Training, Prediction, and Model Evaluation
2. Write unit tests for each function to test that the functions work properly
3. Run the functions and the test file to generate the required results 
and produce a file with logs of any errors and INFO messages

<br/>

### Data Description
The data provided has information about 10,127 of a bank's credit card customers. This includes product information
like 'Credit Limit' and 'Months on book', and background information like 'Marital Status' and 'Customer Age'.

<br/>

### File Descriptions
+ data/ : holds a single csv file with bank data
+ images/ : stores folders with images from EDA and model results
  + eda/ : stores three folders, one with plots for categorical variables, one with plots for numeric variables,
  and one with multivariate plots
  + results/ : holds images of model results, including: classification reports, roc curves, and feature importances
+ logs/ : holds a single file with logs from running churn_script_logging_and_tests.csv
+ models/ : holds model pkl files
+ churn_notebook.ipynb : original code, refactored in churn_library.py
+ churn_library.py : library of functions to find customers who are likely to churn
+ churn_script_logging_and_tests.py : unit tests for all functions in churn_library.py
+ conftest.py : fixtures in the pytest Namespace for use in churn_script_logging_and_tests.py

<br/>

### Run the Files
Two files need to be run in order to complete the project. Both can be run in the command line using the commands below (in the order they appear in)
1. churn_library.py - produces the eda and model results
<br/> `python churn_library.py`
2. churn_script_logging_and_test.py - populates the churn_library.log file by running all unit tests 
<br/> `pytest churn_script_logging_and_tests.py`
