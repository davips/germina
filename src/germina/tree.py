import numpy as np
from pairwiseprediction.combination import pairwise_diff
from scipy.stats import f_oneway


# DecisionTreeRegressor implementation
# Integrating p-value (not sure if p-value use is totally correct)


def calculate_p_value(left, right):
    stat, p_value = f_oneway(left, right)
    return p_value


class DTR:
    def __init__(self, max_depth=None, split_criterion='mse'):
        self.max_depth = max_depth
        self.split_criterion = split_criterion
        self.root = None

    def fit(self, X, y):
        self.root = self._split(X, y, 0)
        return self.root

    def predict(self, X):
        return np.array([self._predict_single(x, self.root) for x in X])

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
                    # best_p_value = self._calculate_p_value(y[left_mask], y[right_mask])

        return best_index, best_value, best_p_value

    def _split_metric(self, left, right):
        if self.split_criterion == 'mse':
            return self._mse(left, right)
        elif self.split_criterion == 'bacc':
            return self._bacc(left, right)
        else:
            raise ValueError("Unsupported split criterion.")

    def _bacc(self, left, right):
        if len(left) == 0 or len(right) == 0:
            return float('inf')
        ma, mb = np.mean(left), np.mean(right)
        diff = pairwise_diff(left.reshape(-1, 1), right.reshape(-1, 1))
        r = np.sum(diff >= 0) / len(diff) if ma >= mb else np.sum(diff < 0) / len(diff)
        return 1-r

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
