import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

class AlwaysOneClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return np.ones((X.shape[0],))

if __name__ == '__main__':
    data = load_iris()
    X, y = data.data, data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    clf = AlwaysOneClassifier()

    clf.fit(X_train, y_train)

    print(clf.score(X_test, y_test))
