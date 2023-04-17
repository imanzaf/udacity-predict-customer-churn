import pytest

def df_plugin():
    return None

def X_train_plugin():
    return None

def X_test_plugin():
    return None

def y_train_plugin():
    return None

def y_test_plugin():
    return None

def lrc_plugin():
    return None

def train_preds_lrc_plugin():
    return None

def test_preds_lrc_plugin():
    return None

def rf_plugin():
    return None

def train_preds_rf_plugin():
    return None

def test_preds_rf_plugin():
    return None


def pytest_configure():
    pytest.df = df_plugin()

    pytest.X_train = X_train_plugin()
    pytest.X_test = X_test_plugin()

    pytest.y_train = y_train_plugin()
    pytest.y_test = y_test_plugin()

    pytest.lrc = lrc_plugin()
    pytest.train_preds_lrc = train_preds_lrc_plugin()
    pytest.test_preds_lrc = test_preds_lrc_plugin()

    pytest.rf = rf_plugin()
    pytest.train_preds_rf = train_preds_rf_plugin()
    pytest.test_preds_rf = test_preds_rf_plugin()