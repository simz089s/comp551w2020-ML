import task1, task2
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt
from scipy.stats import logistic
from scipy.special import expit

def classify_ionosphere(ionosphere_df):
    NUM_ITER = 500
    LEARN_RATE = 0.5
    X = ionosphere_df[2, 0]
    y = ionosphere_df[0]
    
    ones = np.ones(X.shape)
    X = X.to_numpy()
    X = np.hstack((ones, X))
    y = np.reshape(y.to_numpy(), (-1, 1))
    theta = np.array(((0,), (0,)))
    
    model = task2.LogisticRegression(ionosphere_df)
    theta = model.fit(X, y, LEARN_RATE, theta)
    for i in range(NUM_ITER):
        theta = model.fit(X, y, LEARN_RATE, theta)
        if (i % 50) == 0:
            print(model.cost(X, y, theta))

def main():
    df_tuple = task1.clean_data(False)
    ionosphere = df_tuple[0]
    adult = df_tuple[1]
    ttt = df_tuple[2]

    classify_ionosphere(ionosphere)

if __name__ == "__main__":
    main()
