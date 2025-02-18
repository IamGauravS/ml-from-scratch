import numpy as np
from collections import Counter

class KNN:
    def __init__(self, k=3, mode="classification"):
        self.k = k
        self.mode = mode 
        self.X_train = None 
        self.y_train = None 

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y 
    
    def predict(self, X):

        X = np.array(X)
        predictions = [self.predict_single(x) for x in X]
        return np.array(predictions)
    
    def predict_single(self, x):

        distances = np.linalg.norm(self.X_train - x, axis=1)

        k_indices = np.argsort(distances)[:self.k]

        k_nearest_labels = self.y_train[k_indices]

        if self.mode == "classification":
            most_common = Counter(k_nearest_labels).most_common(1)

            return most_common[0][0]
        
        if self.mode == "regression":
            return np.mean(k_nearest_labels)
        
