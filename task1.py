from __future__ import print_function
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

def plot_adult(df):
    title = 'Capital gain per X'
    df.plot(kind='scatter', x='sex', y='capital-gain', title=title, color='lime')

def plot_ionosphere(df):
    df.plot(kind='scatter', x=3, y=4, color='teal')

def plot_ttt(df):
    x_won_with_mm = df.groupby((df.mm == 'x') & (df['x-won'] == 'positive')).count()
    x_won_with_tl = df[(df.tl == 'x') & (df['x-won'] == 'positive')].sum()
    title = 'Number of times X won having taken a specific square'
    x = ('mm', 'tl')
    y = (x_won_with_mm, x_won_with_tl)
    df.plot(kind='bar', x=x, y=y, title=title)

def stats_tests(df_tuple):
    ionosphere = df_tuple[0]
    adult = df_tuple[1]
    ttt = df_tuple[2]

    plot_adult(adult)
    plot_ionosphere(ionosphere)
    plot_ttt(ttt)
    plt.show()

    pass

def main():
    stats_tests(clean_data(False))

if __name__ == "__main__":
    main()
