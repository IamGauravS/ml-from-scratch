import numpy as np 


class LinearRegression:
    def __init__(self, learning_rate = 0.01, epochs = 1000):
        self.weights = None 
        self.bias = None 
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, X, y):

        nSamples, nFeatures = X.shape

        self.weights = np.zero(nFeatures)
        self.bias = 0

        for _ in range(self.epochs):

            linearModel = np.dot(X, self.weights) + self.bias 
            y_pred = self.sigmoid(linearModel)

            dw = (1/nSamples)*np.dot(X.T, (y_pred - y))
            db = (1/nSamples)* np.sum(y_pred - y)

            self.weights -= self.learning_rate * dw 
            self.bias = self.learning_rate * db

    def predict(self, X):
        return np.dot(X, self.weights) + self.bias
       