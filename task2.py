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

    def fit(self, X, y, alpha, theta):
        '''Gradient descent'''
        m = X.shape[0]
        # h = Model.sigmoid(np.matmul(X, theta))
        h = expit(np.matmul(X, theta))
        gradient = np.matmul(X.T, (h - y)) / m
        theta = theta - alpha * gradient
        return theta

    def cost(self, X, y, theta):
        m = X.shape[0]
        # h = Model.sigmoid(np.matmul(X, theta))
        h = expit(np.matmul(X, theta))
        cost = (np.matmul(-y.T, np.log(h)) - np.matmul((1 - y.T), np.log(1 - h))) / m
        return cost

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
