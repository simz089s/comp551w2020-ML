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
    X = ionosphere_df[2]
    y = ionosphere_df[0]

    model = task2.LogisticRegression(ionosphere_df, X.to_numpy(), y.to_numpy(), LEARN_RATE, NUM_ITER)
    model.fit()

def main():
    df_tuple = task1.clean_data(False)
    ionosphere = df_tuple[0]
    adult = df_tuple[1]
    ttt = df_tuple[2]

    classify_ionosphere(ionosphere)

if __name__ == "__main__":
    main()
