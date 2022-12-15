import numpy as np

theta_true_default = np.array([[2.33, 0.67, -1.35]])
epsilon_var_default = 1.0
cov_matrix_default = np.array([[1.0, 2.94],[2.94, 9.0]])

def generate_data(train_sample_size = 10000, test_sample_size = 1000, cov_matrix = cov_matrix_default, theta = theta_true_default, epsilon_var=epsilon_var_default):
    X = np.random.multivariate_normal([0.0, 0.0], cov_matrix, train_sample_size)
    X_test = np.random.multivariate_normal([0.0, 0.0], cov_matrix, test_sample_size)
    
    Xtrain_extended = np.hstack([np.ones((X.shape[0],1)), X])
    y = np.dot(Xtrain_extended, theta.T) + np.random.randn(Xtrain_extended.shape[0],1)*np.sqrt(epsilon_var)
    
    Xtest_extended = np.hstack([np.ones((X_test.shape[0],1)), X_test])
    y_test = np.dot(Xtest_extended, theta.T) + np.random.randn(Xtest_extended.shape[0],1)*np.sqrt(epsilon_var)
    
    return X,y,X_test,y_test



def corr_coeff_2_cov_matr(corr_coeff = 0.0, means = [1.0, 1.0]):
    R = np.array([[1.0, corr_coeff], [corr_coeff, 1.0]])
    c = np.array(means)
    D = np.diag(c)
    cov_matr = np.dot(np.dot(D,R),D)
    
    return cov_matr