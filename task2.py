import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import logistic
from scipy.special import expit

class Model():
    '''
    Base Model class
    '''
    def __init__(self, df):
        """You should use the constructor for the class to initialize the model parameters as attributes, as well as to define other important properties of the model."""
        self.df = df

    @classmethod
    def fit(self, X, y):
        pass

    @classmethod
    def predict(self, X, y_out_pred):
        """TODO: convert probabilities to binary 0/1 using 0.5 as threshold"""
        pass

    @staticmethod
    def evaluate_acc():
        pass

    @staticmethod
    def k_fold_cross_validation():
        pass

    @staticmethod
    def sigmoid(x):
        '''For single values, faster than scipy.stats.logistic or even scipy.special.expit'''
        return 1 / (1 + np.exp(-x))


class LogisticRegression(Model):
    '''Logistic regression using full batch gradient descent'''
    def __init__(self, df):
        super(LogisticRegression, self).__init__(df)

    def fit(self, X, y, lr=.01, max_num_iter=500, decay=.96, decay_rate=50, eps=1e-2, min_lr=.001):
        # Prepend bias terms to X
        ones = np.ones(X.shape)
        X = np.hstack((ones, X))
        N, D = X.shape

        # Transpose y vector into column for matrix multiplication
        # y needs 2 values for each X because of bias, same for theta
        y = np.reshape(y, (-1, 1))
        w = np.zeros((D, 1))

        g = np.inf

        lr_0 = lr
        for i in range(max_num_iter):
            if (np.linalg.norm(g) <= eps) or (lr < min_lr):
                return w
            w = self.gradient_descent(X, y, w, lr)
            lr = lr_0 * (decay**np.floor(i / decay_rate))
            if (i % 100) == 0:
                '''Print cost every enth iterations'''
                J = self.cost(X, y, w)
                print(f'{w}, {J}, {lr}')
        
        return w

    def gradient_descent(self, X, y, w, alpha):
        '''One step of batch gradient descent'''
        N, D = X.shape
        # z_logit = np.dot(w.T, X)
        # yh = Model.sigmoid(z_logit) # Activation (logistic) function sigma
        yh = expit(np.matmul(X, w))
        grad = np.matmul(X.T, (yh - y)) / N
        w = w - alpha * grad
        return w

    def cost(self, X, y, w):
        # z = Model.sigmoid(np.matmul(X, w))
        z = expit(np.matmul(X, w))
        J = np.mean( np.matmul(y.T, np.log1p(z)) + np.matmul((1 - y.T), np.log1p(np.exp(z))) )
        return J

    def predict(self, y_out_pred):
        pass


class NaiveBayes(Model):
    '''Naïve Bayes'''
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self, y_out_pred):
        pass
