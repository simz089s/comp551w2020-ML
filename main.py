import task1, task2
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import logistic
from scipy.special import expit

def classify_ionosphere(ionosphere_df):
    LEARN_RATE = 0.5
    NUM_ITER = 500
    DECAY = 0.96
    DECAY_RATE = 50
    X = ionosphere_df.iloc[:, 2:4]
    y = ionosphere_df.iloc[:, -1]

    # good = ionosphere_df.loc[y == 'g']
    # bad = ionosphere_df.loc[y == 'b']
    # ax = plt.gca()
    # good.plot(x=2, y=3, kind='scatter', label='Good', ax=ax, color='lime')
    # bad.plot(x=2, y=3, kind='scatter', label='Bad', ax=ax, color='red')
    # plt.show()

    # g->1, b->0
    model = task2.LogisticRegression(ionosphere_df, X.to_numpy(), np.vectorize({'b':0, 'g':1}.get)(y), LEARN_RATE, NUM_ITER, DECAY, DECAY_RATE)
    model.fit()

def main():
    cleaned_dfs = task1.clean_data(False)
    ionosphere = cleaned_dfs[0]
    # adult = cleaned_dfs[1]
    # ttt = cleaned_dfs[2]

    classify_ionosphere(ionosphere)

if __name__ == "__main__":
    main()
