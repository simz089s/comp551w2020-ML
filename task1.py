from __future__ import print_function
import numpy as np
import pandas as pd

def read_and_print_csv(*args):
    for filename in args:
        print(f"### {filename} ###")
        df = pd.read_csv(filename, sep=',', header=None)
        print(df.values)

def main():
    # read_and_print_csv("adult.all", "ionosphere.data")
    adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-exceeds']
    adult_df = pd.read_csv("adult.all", usecols=adult_cols)[adult_cols]
    print(adult_df.values)
    adult_df.to_csv("adult.clean.data")
    return 0

if __name__ == '__main__':
    main()
