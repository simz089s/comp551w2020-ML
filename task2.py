class Model():
    '''
    Base Model class
    '''

    def init(self):
        """You should use the constructor for the class to initialize the modelparameters as attributes, as well as to define other important properties of the model."""
        pass

    def fit(self, X, y, lr, gdi):
        """Use gradient descent for optimization"""
        pass

    def predict(self, X, yhat):
        """TODO: convert probabilities to binary 0/1 using 0.5 as threshold"""
        pass


class LogisticRegression(Model):
    '''Logistic regression using full batch gradient descent'''

    def init(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


class NaiveBayes(Model):
    '''Naïve Bayes'''

    def init(self):
        pass

    def fit(self):
        pass

    def predict(self):
        pass


def evaluate_acc():
    pass

def k_fold_cross_validation():
    pass
