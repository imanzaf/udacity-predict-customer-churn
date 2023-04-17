'''
This file defines fixture in the Namespace,
    to make them accessible and editable in other files.

Author: Iman Zafar
Date: April 2023
'''

import pytest


# configure function to set up empty variables
def pytest_configure():
    pytest.df = None

    pytest.X_train = None
    pytest.X_test = None

    pytest.y_train = None
    pytest.y_test = None

    pytest.lrc = None
    pytest.train_preds_lrc = None
    pytest.test_preds_lrc = None

    pytest.rf = None
    pytest.train_preds_rf = None
    pytest.test_preds_rf = None