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

    def fit(self, X, y, lr, max_num_iter, decay, decay_rate):
        # Prepend bias terms to X
        ones = np.ones(X.shape)
        X = np.hstack((ones, X))

        # Transpose y vector into column for matrix multiplication
        # y needs 2 values for each X because of bias, same for theta
        y = np.reshape(y, (-1, 1))
        # w = np.array(((0,), (0,)))
        w = np.zeros((X.shape[1], 1))

        cur_lr = lr
        for i in range(max_num_iter):
            w = self.gradient_descent(X, y, w, cur_lr)
            if (i % 50) == 0:
                '''Print cost every 50 iterations'''
                J = self.cost(X, y, w)
                print(f'{w}, {J}, {cur_lr}')
            if cur_lr < 0.001:
                return w
            cur_lr = lr * (decay**np.floor(i / decay_rate))

    def gradient_descent(self, X, y, w, alpha):
        '''One step of batch gradient descent'''
        N, D = X.shape
        # h = Model.sigmoid(np.matmul(X, w))
        h = expit(np.matmul(X, w)) # Activation (logistic) function sigma
        # logit = np.dot(w.T, X)
        grad = np.matmul(X.T, (h - y)) / N
        w = w - alpha * grad
        return w

    def cost(self, X, y, w):
        # h = Model.sigmoid(np.matmul(X, w))
        # h = expit(np.matmul(X, w))
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
