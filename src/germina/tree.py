import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeRegressor


class DTR(DecisionTreeRegressor):

    def __init__(self, max_depth=None):
        self.max_depth = max_depth
        self._estimator = False

    def bacc(self, a, b):
        if len(a) * len(b) == 0:
            return 0
        ma, mb = np.mean(a), np.mean(b)
        expected = a > b


        predicted = ma >= mb
        return balanced_accuracy_score(expected, predicted)

    def _best_split(self, X, y):
        best_score = float('inf')
        best_value = None
        best_index = None
        n_features = X.shape[1]
        for feature_idx in range(n_features):
            for value in set(X[:, feature_idx]):
                left_mask = X[:, feature_idx] <= value
                right_mask = X[:, feature_idx] > value
                score = self.bacc(y[left_mask], y[right_mask])
                if score < best_score:
                    best_score = score
                    best_value = value
                    best_index = feature_idx
        return best_index, best_value

    def _best_split__original(self, X, y):
        best_mse = float('inf')
        best_value = None
        best_index = None
        n_features = X.shape[1]

        for feature_idx in range(n_features):
            for value in set(X[:, feature_idx]):
                left_mask = X[:, feature_idx] <= value
                right_mask = X[:, feature_idx] > value
                mse = self._mse(y[left_mask], y[right_mask])

                if mse < best_mse:
                    best_mse = mse
                    best_value = value
                    best_index = feature_idx

        return best_index, best_value

    def _mse(self, left, right):
        total_samples = len(left) + len(right)
        mse_left = self._mean_squared_error(left)
        mse_right = self._mean_squared_error(right)
        return (len(left) / total_samples) * mse_left + (len(right) / total_samples) * mse_right

    def _mean_squared_error(self, y):
        if len(y) == 0:
            return 0
        mean = np.mean(y)
        return np.mean((y - mean) ** 2)

    def _terminal_node(self, y):
        return np.mean(y)

    def _split(self, X, y, depth):
        if len(y) == 0 or depth == self.max_depth:
            return self._terminal_node(y)

        feature_idx, value = self._best_split(X, y)
        left_mask = X[:, feature_idx] <= value
        right_mask = X[:, feature_idx] > value

        left = self._split(X[left_mask], y[left_mask], depth + 1)
        right = self._split(X[right_mask], y[right_mask], depth + 1)

        return (feature_idx, value, left, right)

    def fit(self, X, y):
        self._estimator = True
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        if not isinstance(y, np.ndarray):
            y = np.array(y)
        self.root = self._split(X, y, 0)

    def predict(self, X):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        return np.array([self._predict_single(x) for x in X])

    def _predict_single(self, x):
        node = self.root
        while isinstance(node, tuple):
            feature_idx, value, left, right = node
            if x[feature_idx] <= value:
                node = left
            else:
                node = right
        return node

    def __sklearn_is_fitted__(self):
        return self._estimator

    # from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import pandas as pd
# data_url = "http://lib.stat.cmu.edu/datasets/boston"
# raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
# X = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
# y = raw_df.values[1::2, 2]
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
#
# dtr = DTR(max_depth=4)
# dtr.fit(X_train, y_train)
#
# y_pred = dtr.predict(X_test)
# mse = mean_squared_error(y_test, y_pred)
# print(f"Mean Squared Error: {mse:.3f}")
