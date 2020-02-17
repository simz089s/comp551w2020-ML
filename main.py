import task1, task2
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import logistic
from scipy.special import expit

def classify_ionosphere(ionosphere_df):
    LEARN_RATE = .1
    MAX_NUM_ITER = 8000
    DECAY = .96
    DECAY_RATE = 50
    EPS = 1e-2
    REGUL_LAMBDA = .1
    k = 5 # of folds for cross validation

    X_df = ionosphere_df.iloc[:, :-1]
    y_df = ionosphere_df.iloc[:, -1]
    X = X_df.to_numpy()
    y = np.vectorize({'b':0, 'g':1}.get)(y_df)

    model = task2.LogisticRegression(ionosphere_df)
    # b(ad)->0, g(ood)->1
    w = model.fit(X, y, LEARN_RATE, MAX_NUM_ITER, DECAY, DECAY_RATE, EPS, REGUL_LAMBDA)
    print(w)
    yh = model.predict(X, w)
    acc = model.eval_acc(X, y, yh, w)
    print(f'Accuracy: {acc} %')
    perfs = model.kfold_crossval(X, y, k)
    print(f'Cross validation: {perfs}')

    # good = ionosphere_df.loc[y_df == 'g']
    # bad = ionosphere_df.loc[y_df == 'b']
    # ax = plt.gca()
    # good.plot(x=2, y=3, kind='scatter', label='Good', ax=ax, color='lime')
    # bad.plot(x=2, y=3, kind='scatter', label='Bad', ax=ax, color='red')
    # plt.show()


def classify_adult(adult_df):
    y = adult_df.iloc[:, 14]
    rich = adult_df.loc[y == '>50K']
    poor = adult_df.loc[y == '<=50K']
    cat_columns = [1, 3, 5, 6, 7, 8, 9, 13]
    cont_colums = [0, 2, 4, 10, 11, 12]
    modeling = task2.NaiveBayes(cat_columns, cont_colums)
    modeling.fit(rich, poor)

    print(modeling.predict(adult_df.iloc[8, :]))
def main():
    cleaned_dfs = task1.clean_data(False)
    ionosphere = cleaned_dfs[0]
    adult = cleaned_dfs[1]
    # ttt = cleaned_dfs[2]

    classify_ionosphere(ionosphere)
    classify_adult(adult)

if __name__ == "__main__":
    main()
