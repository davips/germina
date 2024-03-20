import numpy as np
from sklearn.metrics import balanced_accuracy_score
from sklearn.tree import DecisionTreeRegressor

# DecisionTreeRegressor implementation
# Integrating p-value (not sure if p-value use is totally correct)

import numpy as np
from scipy.stats import f_oneway


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


# Implementation of tree to mermaid

class DTR2:
    def __init__(self, max_depth=None, split_criterion='mse'):
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._split(X, y, 0)
        return self.root

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

    def _calculate_p_value(self, left, right):
        stat, p_value = f_oneway(left, right)
        return p_value

    def _split(self, X, y, depth):
        if len(y) == 0 or depth == self.max_depth:
            return np.mean(y)

        feature_idx, split_value, p_value = self._best_split(X, y)
        if feature_idx is None:
            return np.mean(y)

        left_mask = X[:, feature_idx] <= split_value
        right_mask = X[:, feature_idx] > split_value

        left = self._split(X[left_mask], y[left_mask], depth + 1)
        right = self._split(X[right_mask], y[right_mask], depth + 1)

        return {'index': feature_idx, 'value': split_value, 'p_value': p_value, 'left': left, 'right': right}

    def _best_split(self, X, y):
        best_metric = float('inf')
        best_index = None
        best_value = None
        best_p_value = None
        for feature_idx in range(X.shape[1]):
            for split_value in np.unique(X[:, feature_idx]):
                left_mask = X[:, feature_idx] <= split_value
                right_mask = X[:, feature_idx] > split_value
                metric = self._split_metric(y[left_mask], y[right_mask])

                if metric < best_metric:
                    best_metric = metric
                    best_index = feature_idx
                    best_value = split_value
                    best_p_value = self._calculate_p_value(y[left_mask], y[right_mask])

        return best_index, best_value, best_p_value

    def _split_metric(self, left, right):
        if self.split_criterion == 'mse':
            return self._mse(left, right)
        # ----
        # Add your split method
        elif self.split_citerion == 'awesome_split':
            return self._awesome_split(left, right)
        # ----
        else:
            raise ValueError("Unsupported split criterion.")

    def _awesome_split(left, right):
        if len(left) == 0 or len(right) == 0:
            return float('inf')
        # Describe your split method
        return True

    def _mse(self, left, right):
        if len(left) == 0 or len(right) == 0:
            return float('inf')
        return np.mean((left - np.mean(left)) ** 2) + np.mean((right - np.mean(right)) ** 2)

    def _predict_single(self, x, node):
        if not isinstance(node, dict):
            return node

        if x[node['index']] <= node['value']:
            return self._predict_single(x, node['left'])
        else:
            return self._predict_single(x, node['right'])


def tree_to_mermaid(dtree, var_labels, node_label='A', node_number=0):
    """
    Convert a decision tree expressed as a dictionary into a mermaid graph
    Run the printed output into a mermaid reader

    labels = [f'Feature {i}' for i in range(len(X))]


    mermaid_graph = "graph TD\n" + tree_to_mermaid(model, labels)
    print(mermaid_graph)

    """

    mermaid_str = ''

    # Leaf node check
    if isinstance(dtree, (int, float)):
        return f'{node_label}["Prediction: {dtree:.4f}"]\n'

    # Generating unique labels: e.g. A[ ], B[ ] etc
    def generate_label(node_number):
        label = ''
        while True:
            node_number, remainder = divmod(node_number, 26)
            remainder += 65
            label = chr(remainder) + label
            if node_number == 0:
                break
            node_number -= 1
        return label

    left_number = 2 * node_number + 1
    right_number = 2 * node_number + 2
    left_label = generate_label(left_number)
    right_label = generate_label(right_number)

    mermaid_str += f'{node_label}[ ]\n'

    left_child = dtree.get('left')
    if left_child is not None:
        decision_text = f'{var_labels[dtree["index"]]} > {dtree["value"]:.4f}'
        mermaid_str += f'{node_label} --"{decision_text}"--> {left_label}\n'
        mermaid_str += tree_to_mermaid(left_child, var_labels, left_label, left_number)

    right_child = dtree.get('right')
    if right_child is not None:
        decision_text = f'{var_labels[dtree["index"]]} <= {dtree["value"]:.4f}'
        mermaid_str += f'{node_label} --"{decision_text}"--> {right_label}\n'
        mermaid_str += tree_to_mermaid(right_child, var_labels, right_label, right_number)

    return mermaid_str
