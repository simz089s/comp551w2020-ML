import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

def clean_data(save_to_csv):
    """TODO: remove ionosphere headers??"""
    NUM_COLS_IONOSPHERE = 35
    ionosphere_cols = list(range(NUM_COLS_IONOSPHERE))
    ionosphere_df = pd.read_csv("ionosphere.data", names=ionosphere_cols, usecols=ionosphere_cols)[ionosphere_cols]
    # ionosphere_df.dropna(how='any', inplace=True)
    # if save_to_csv:
    #     ionosphere_df.to_csv("ionosphere.clean.data", index=False, header=False)

    adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-exceeds']
    adult_df = pd.read_csv("adult.all", names=adult_cols, usecols=adult_cols, na_values=['?'], skipinitialspace=True)[adult_cols]
    adult_df.dropna(how='any', inplace=True)
    if save_to_csv:
        adult_df.to_csv("adult.clean.data", index=False, header=False)
    
    ttt_cols = ['tl', 'tm', 'tr', 'ml', 'mm', 'mr', 'bl', 'bm', 'br', 'x-won']
    ttt_df = pd.read_csv("tic-tac-toe.data", names=ttt_cols, usecols=ttt_cols)[ttt_cols]

    return (ionosphere_df, adult_df, ttt_df)

def main():
    clean_data(False)

if __name__ == "__main__":
    main()
