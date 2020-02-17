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
    def eval_acc():
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

    def fit(self, X, y, lr=.01, max_num_iter=500, decay=.96, decay_rate=50, eps=1e-2, min_lr=.001, regul_lambda=0):
        # Prepend bias terms to X
        # ones = np.ones(X.shape)
        # X = np.hstack((ones, X))
        N, D = X.shape

        # Make y a column vector for matrix multiplication
        y = np.reshape(y, (-1, 1))
        w = np.zeros((D, 1))

        g = np.inf

        J = self.cost(X, y, w)
        print(f'Initial cost: {J}\nInitial learning rate: {lr}')
        lr_0 = lr
        for i in range(max_num_iter):
            if (np.linalg.norm(g) <= eps) or (lr < min_lr):
                break
            g = self.gradient_descent(X, y, w, lr)
            w = g
            lr = lr_0 * (decay**np.floor(i / decay_rate))

        J = self.cost(X, y, w)
        print(f'Final cost: {J}\nFinal learning rate: {lr}')
        
        return w

    def gradient_descent(self, X, y, w, alpha, lambda_=0):
        '''One step of batch gradient descent'''
        N, D = X.shape
        # z_logit = np.dot(w.T, X)
        # yh = Model.sigmoid(z_logit) # Activation (logistic) function sigma
        # yh = expit(np.matmul(X, w))
        yh = expit(X @ w)
        grad = np.matmul(X.T, (yh - y)) / N
        grad[1:] += lambda_ * np.sign(w[1:])
        w = w - alpha * grad
        return w

    def cost(self, X, y, w):
        z = expit(np.matmul(X, w))
        J = np.mean( np.matmul(y.T, np.log1p(z)) + np.matmul((1 - y.T), np.log1p(np.exp(z))) )
        return J

    def predict(self, X, w):
        yh = (np.matmul(X, w) > 0.5).astype(int)
        return yh

    def eval_acc(self, X, y, yh, w):
        N, D = X.shape
        y = np.reshape(y, (-1, 1))
        correct = (yh == y)
        acc = (np.sum(correct) / N) * 100
        return acc

    def kfold_crossval(self, X, y, k=5):
        perfs = []
        X_folds = np.array_split(X, k, axis=0)
        y_folds = np.array_split(y, k, axis=0)
        for i, X_fold in enumerate(X_folds):
            X_test_set = X_fold
            X_sets_before = X_folds[:i]
            X_sets_after = X_folds[i+1:]
            y_sets_before = y_folds[:i]
            y_sets_after = y_folds[i+1:]
            X_train_set = np.concatenate(X_sets_before + X_sets_after, axis=0)
            y_train_set = np.concatenate(y_sets_before + y_sets_after, axis=0)
            w = self.fit(X_train_set, y_train_set)
            yh = self.predict(X_train_set, w)
            acc = self.eval_acc(X_test_set, y_train_set, yh, w)
            perfs.append(acc)
        return perfs


class NaiveBayes(Model):
    '''Naïve Bayes'''
    def __init__(self):
        pass

    def fit(self):
        pass

    def predict(self, y_out_pred):
        pass
