import numpy as np
from sample_bootstrap import *
from linear_regression_model import *

def estimate_parameters_range(X, y):
    B = 1000
    Nb = X.shape[0]
    theta_values_curr_sample_size = []
    for i in range(B):
        Xtrain,ytrain = sample_bootstrap(X,y, sample_size=Nb)
        lr = linear_regression()
        lr.fit(Xtrain,ytrain)
        theta_values_curr_sample_size.append(np.copy(lr.theta))
    theta_values_curr_sample_size = [t[:,:,np.newaxis] for t in theta_values_curr_sample_size]
    theta_values_curr_sample_size = np.concatenate(theta_values_curr_sample_size, axis=-1)
    theta_means = np.mean(theta_values_curr_sample_size, axis=-1)
    theta_min = np.min(theta_values_curr_sample_size, axis=-1)
    theta_max = np.max(theta_values_curr_sample_size, axis=-1)
    theta_range = theta_max - theta_min
    # t1_mean = np.mean(theta_values_curr_sample_size[:,1])
    # t1_min = np.min(theta_values_curr_sample_size[:,1])
    # t1_max = np.max(theta_values_curr_sample_size[:,1])
    # t1_range = (t1_max-t1_min)
    # t1_linspace = np.linspace(t1_mean-10*t1_range, t1_mean+10*t1_range, 200)
    #
    # t2_mean = np.mean(theta_values_curr_sample_size[:,2])
    # t2_min = np.min(theta_values_curr_sample_size[:,2])
    # t2_max = np.max(theta_values_curr_sample_size[:,2])
    # t2_range = (t2_max-t2_min)
    # t2_linspace = np.linspace(t2_mean-10*t2_range, t2_mean+10*t2_range, 200)
    #
    # t0_mean = np.mean(theta_values_curr_sample_size[:,0])
    
    # return theta_values_curr_sample_size, t1_linspace, t2_linspace, t0_mean
    return theta_means, theta_min, theta_max