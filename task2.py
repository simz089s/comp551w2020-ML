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
    def __init__(self, df, X, y, learn_rate, num_iter, decay, decay_rate):
        self.X = X
        self.y = y
        self.learn_rate = learn_rate
        self.num_iter = num_iter
        self.decay = decay
        self.decay_rate = decay_rate
        super(LogisticRegression, self).__init__(df)

    def fit(self):
        # Prepend bias terms to X
        self.ones = np.ones(self.X.shape)
        self.X = np.hstack((self.ones, self.X))

        # Transpose y vector into column for matrix multiplication
        # y needs 2 values for each X because of bias, same for theta
        self.y = np.reshape(self.y, (-1, 1))
        # self.theta = np.array(((0,), (0,)))
        self.theta = np.zeros((self.X.shape[1], 1))

        # self.N = self.X.shape[0]
        # self.h = expit(np.matmul(self.X, self.theta))
        # self.z = np.dot(self.X, self.theta)
        
        curr_learn_rate = self.learn_rate
        for i in range(self.num_iter):
            self.theta = self.gradient_descent(self.X, self.y, curr_learn_rate, self.theta)
            if (i % 50) == 0:
                '''Print cost every 50 iterations'''
                J = self.cost(self.X, self.y, self.theta)
                print(f'{self.theta}, {J}, {curr_learn_rate}')
            if curr_learn_rate < 0.001:
                return self.theta
            curr_learn_rate = self.learn_rate * (self.decay**np.floor(i / self.decay_rate))

    def gradient_descent(self, X, y, alpha, theta):
        '''One step of full batch gradient descent'''
        N = X.shape[0]
        # h = Model.sigmoid(np.matmul(X, theta))
        h = expit(np.matmul(X, theta)) # Activation (logistic) function sigma
        # logit = np.dot(theta.T, X)
        gradient = np.matmul(X.T, (h - y)) / N
        theta = theta - alpha * gradient
        return theta

    def cost(self, X, y, w):
        # h = Model.sigmoid(np.matmul(X, theta))
        # h = expit(np.matmul(X, theta))
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
