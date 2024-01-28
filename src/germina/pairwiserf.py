import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.utils.multiclass import unique_labels

from germina.pairwise import pairwise_diff, pairwise_hstack


class PairwiseRF(BaseEstimator, RegressorMixin):
    """
    >>> import numpy as np
    >>> X = np.array([[1,2], [2,3], [3,4], [4,5]])
    >>> y = np.array([[3], [5], [7], [9]])
    >>> prf = PairwiseRF()
    >>> prf.fit(X, y)
    PWRandomForestClassifier()
    >>> prf.predict(X)
    array([3., 5., 7., 9.])
    >>> prf = PairwiseRF(diff=False)
    >>> prf.fit(X, y)
    PWRandomForestClassifier()
    >>> prf.predict(X)
    array([3., 5., 7., 9.])
    """

    def __init__(self, n_estimators=100,
                 # criterion="gini", max_depth=None, min_samples_split=2, min_samples_leaf=1,
                 # min_weight_fraction_leaf=0.0, max_features="sqrt", max_leaf_nodes=None, min_impurity_decrease=0.0,
                 # bootstrap=True, oob_score=False,
                 n_jobs=None, random_state=None,
                 # verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None,
                 diff=True, nested="RFc"):
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.diff = diff
        self.nested = nested
        # TODO: complete all RF args here
        if nested == "RFc":
            self.rf = RandomForestClassifier(self.n_estimators, n_jobs=self.n_jobs, random_state=self.random_state)
        else:
            self.rf = RandomForestRegressor(self.n_estimators, n_jobs=self.n_jobs, random_state=self.random_state)

    # def get_params(self, deep=False):
    #     return {}  # "n_estimators": self.n_estimators, "n_jobs": self.n_jobs, "random_state": self.random_state, "diff": self.diff}

    def fit(self, X, y):
        if not isinstance(X, np.ndarray):
            X = np.array(X)
            y = np.array(y)
        # self.classes_ = unique_labels(y)
        self.idxs = np.argsort(y.flatten(), kind="stable").flatten()
        self.Xtr = X[self.idxs]
        self.ytr = y[self.idxs]
        self.mn, self.mx = self.ytr.min(), self.ytr.max()
        self.std = y.std()
        Mtr = np.hstack([X, y.reshape(len(y), -1)])
        D = pairwise_diff(Mtr, Mtr) if self.diff else pairwise_hstack(Mtr, Mtr, handle_last_as_y=True)
        ytr_rf = D[:, -1]
        if self.nested == "RFc":
            D[np.abs(ytr_rf) < self.std / 2, -1] = 0
            ytr_rf = np.sign(ytr_rf).astype(int)
        Xtr_rf = D[:, :-1]
        # print("internal X:", Xtr_rf.shape)
        self.rf.fit(Xtr_rf, ytr_rf)
        return self

    def predict(self, X):
        # check_is_fitted(self)
        if not isinstance(X, np.ndarray):
            X = np.array(X)
        l = []
        for i in range(X.shape[0]):
            x = X[i:i + 1, :]
            Xts_rf = pairwise_diff(x, self.Xtr) if self.diff else pairwise_hstack(x, self.Xtr)
            # Xts_rf = np.vstack([pairwise_diff(x, self.Xtr) if self.diff else pairwise_hstack(x, self.Xtr),
            #                pairwise_diff(self.Xtr, x) if self.diff else pairwise_hstack(self.Xtr, x)])
            zts_rf = self.rf.predict(Xts_rf)  # reminder: already sorted by ytr
            if self.nested == "RFc":
                acc = np.cumsum(zts_rf)
                mx_mask = acc == np.max(acc)
                mx_idxs = np.flatnonzero(mx_mask)
                candidates = self.ytr[mx_idxs]
                zeromxs = self.ytr[(zts_rf == 0) & mx_mask]
                if zeromxs.shape[0] > 0:
                    candidates = zeromxs
            else:
                candidates = self.ytr + zts_rf
            v = np.mean(candidates)
            l.append(v)
        return np.array(l)

    def __repr__(self, **kwargs):
        return "PW" + repr(self.rf)

    # def __sklearn_clone__(self):
    #     return PairwiseRF(self.n_estimators, self.n_jobs, self.random_state, self.diff)
