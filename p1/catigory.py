import numpy as np
import scipy as sp
import pandas as pd
import math
from matplotlib import pyplot as plt
from scipy.stats import logistic
from scipy.special import expit


class catigory():

    def __init__(self, cat, prob):
        self.cat = cat  # the name of the catigory
        self.prob = prob # the probability of it

    def getcat(self):
        return self.cat

    def getprob(self):
        return self.prob
