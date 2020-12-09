from linear_regression_model import *
import numpy as np


def RMSE(y_true, y_pred):
    return np.sqrt(np.sum(np.square(np.squeeze(np.asarray(y_true)) - np.squeeze(np.asarray(y_pred))))/y_pred.shape[0])


def loss(X_val,y_val,theta):
    lr = linear_regression()
    # lr.theta = theta.reshape(X_val.shape[1]+1, 1)
    lr.theta = theta
    y_pred = lr.predict(X_val)
    l = RMSE(y_val, y_pred)
    return l