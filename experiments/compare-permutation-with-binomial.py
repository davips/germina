import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.dummy import DummyClassifier
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from sklearn.model_selection import cross_val_score, permutation_test_score
from sklearn.tree import DecisionTreeClassifier

from germina.stats import p_value

# Load the breast cancer dataset
breast_cancer = load_breast_cancer()
X = breast_cancer.data
y = breast_cancer.target
rnd = np.random.default_rng(0)
rnd.shuffle(X)
rnd = np.random.default_rng(0)
rnd.shuffle(y)
n = 200
X = X[:n]
y = y[:n]
print(len(X))
# Define your model
# model = DummyClassifier(random_state=0)
model = DecisionTreeClassifier(random_state=0)

# Perform leave-one-out cross-validation
loo = LeaveOneOut()
preds = cross_val_predict(model, X, y, cv=loo)

# Calculate the observed accuracy
observed_accuracy = accuracy_score(y, preds)
print(f"{p_value(observed_accuracy, n)=:.6f}")

# Perform permutation test
num_permutations = n
score, permutation_scores, p = permutation_test_score(model, X, y, cv=loo, n_permutations=num_permutations, scoring="accuracy", n_jobs=-1)

print("Observed Accuracy:", observed_accuracy)
print(f"{p:.4f}")
