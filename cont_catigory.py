import numpy as np
import scipy as sp
import pandas as pd
from math import sqrt
from math import pi
from math import exp
from matplotlib import pyplot as plt
from scipy.stats import logistic
from scipy.special import expit


class cont_catigory():

    def __init__(self, mean, standard_d):
        self.mean = mean
        self.standard_d = standard_d


    def get_prob(self, input):
        exponent = exp(-((input - self.mean) ** 2 / (2 * self.standard_d ** 2)))
        return (1 / (sqrt(2 * pi) * self.standard_d)) * exponent
