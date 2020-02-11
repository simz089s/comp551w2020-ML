from __future__ import print_function
import numpy as np
import scipy as sp

def main():
    adult_cols = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income-exceeds']
    adult_data = np.genfromtxt("adult.all", delimiter=',', dtype='unicode', encoding=None, names=adult_cols, autostrip=True, missing_values='?')
    print(adult_data[32565])
    np.logical_and(adult_data == '?')
    # adult_data = adult_data[~np.any(np.logical_and(np.isnan(adult_data), adult_data == '?'), axis=0)]
    # print(adult_data[32565])
    # np.savetxt("adult.clean.data", adult_data, delimiter=", ", fmt="%s")
    return 0

if __name__ == '__main__':
    main()
