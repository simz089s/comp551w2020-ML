from __future__ import print_function
import numpy as np
import scipy as sp
import pandas as pd
from matplotlib import pyplot as plt

def clean_data(save_to_csv):
    """TODO: remove ionosphere headers??"""
    ionosphere_cols = list(range(35))
    ionosphere_df = pd.read_csv("ionosphere.data", names=ionosphere_cols, usecols=ionosphere_cols)[ionosphere_cols]
    ionosphere_df.dropna(how='any', inplace=True)
    if save_to_csv:
        ionosphere_df.to_csv("ionosphere.clean.data", index=False, header=False)

    adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-exceeds']
    adult_df = pd.read_csv("adult.all", names=adult_cols, usecols=adult_cols, na_values=['?'], skipinitialspace=True)[adult_cols]
    adult_df.dropna(how='any', inplace=True)
    if save_to_csv:
        adult_df.to_csv("adult.clean.data", index=False, header=False)

    return (ionosphere_df, adult_df)

def stats_tests(df_tuple):
    ionosphere = df_tuple[0]
    print(ionosphere)
    adult = df_tuple[1]
    # plt.scatter('x', 'y', data=ionosphere)
    # plt.plot(ionosphere)
    # plt.imshow(ionosphere)
    ionosphere.plot(kind='scatter', x=3, y=4, color='teal')
    plt.show()

def main():
    stats_tests(clean_data(False))
    return 0

if __name__ == '__main__':
    main()
