import numpy as np
class LogisticRegression:
    def __init__(self, learning_rate = 0.01, epochs = 100):
        self.learning_rate =  learning_rate 
        self.epochs = epochs 
        self.weights = None 
        self.bias = None 

    def sigmoid(self, z):
        return 1/ (1 + np.exp(-z))