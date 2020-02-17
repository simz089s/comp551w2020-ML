import catigory, cont_catigory
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
    def __init__(self, catigorical_cols, continious_cols):
        self.catigorical_cols=catigorical_cols
        self.continious_cols=continious_cols



    def fit(self, data_positive, data_negitive):
        #classify = data.iloc[:, 14]
        #rich = data.loc[y == '>50K']
        #poor = data.loc[y == '<=50K']

        #pos first
        total_num_pos=len(data_positive)
        self.probabilites_pos = []
        for i in self.catigorical_cols:
            colum_list = []
            catigories = data_positive.iloc[:, i].unique()
            col = data_positive.iloc[:, i]
            for cat in catigories:
                data_in_catagor = data_positive.loc[col == cat]
                prob = len(data_in_catagor)/total_num_pos
                new_entry = catigory.catigory(cat, prob)
                colum_list.append(new_entry)

            self.probabilites_pos.append(colum_list)



        total_num_neg = len(data_negitive)
        self.probabilites_neg = []
        for i in self.catigorical_cols:
            colum_list = []
            catigories = data_negitive.iloc[:, i].unique()
            col = data_negitive.iloc[:, i]
            for cat in catigories:
                data_in_catagor = data_negitive.loc[col == cat]
                prob = len(data_in_catagor) / total_num_neg
                new_entry = catigory.catigory(cat, prob)
                colum_list.append(new_entry)
            self.probabilites_neg.append(colum_list)

        self.prob_pos = total_num_pos/(total_num_pos+total_num_neg)
        self.prob_neg = total_num_neg / (total_num_pos + total_num_neg)

        self.continious_prob_pos = []
        for i in self.continious_cols:
            total = data_positive.iloc[:, i].sum()
            mean = total/len(data_positive.iloc[:, i])
            stdev = data_positive.iloc[:, i].std()
            new_ent = cont_catigory.cont_catigory(mean, stdev)
            self.continious_prob_pos.append(new_ent)

        self.continious_prob_neg = []
        for i in self.continious_cols:
            total = data_negitive.iloc[:, i].sum()
            mean = total / len(data_negitive.iloc[:, i])
            stdev = data_negitive.iloc[:, i].std()
            new_ent = cont_catigory.cont_catigory(mean, stdev)
            self.continious_prob_neg.append(new_ent)

    def predict(self, input_point):
        yYes=self.prob_pos
        pno=self.prob_neg
        #print(self.probabilites_pos)
        #print(self.catigorical_cols)
        #print(len(self.probabilites_pos))
        #print(len(self.catigorical_cols))
        count = 0
        for i in self.catigorical_cols:
            cata = input_point.iloc[i]
            for j in self.probabilites_pos[count]:
                if j.getcat()==cata:
                    yYes= yYes*j.getprob()
            count = count+1

        #print(len(self.continious_cols))
        #print(len(self.continious_prob_pos))
        #print(len(input_point))
        count = 0
        for i in self.continious_cols:
            yYes = yYes * self.continious_prob_pos[count].get_prob(input_point.iloc[i])
            count=count+1

        count = 0
        for i in self.catigorical_cols:
            cata = input_point.iloc[i]
            for j in self.probabilites_neg[count]:
                if j.getcat()==cata:
                    pno= pno*j.getprob()
            count=count+1
        count = 0
        for i in self.continious_cols:
            pno = pno * self.continious_prob_neg[count].get_prob(input_point.iloc[i])
            count=count+1


        pYesNormal=(yYes)/(yYes+pno)
        pNoNormal = (pno)/(yYes+pno)

        if pYesNormal>pNoNormal:
            perdiction=True
        else:
            perdiction=False

        return perdiction
