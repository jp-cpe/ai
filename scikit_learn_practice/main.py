import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin

class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, x, y):
        return self
    
    def predict(self, x):
        return np.ones((x.shape[0],))