import warnings

import dalex as dx
import numpy as np
from indexed import Dict
from joblib.parallel import Parallel, delayed
from lange import ap
from lightgbm import LGBMClassifier as LGBMc
from lightgbm import LGBMRegressor as LGBMr
from pairwiseprediction.classifier import PairwiseClassifier
from pairwiseprediction.combination import pairwise_diff, pairwise_hstack
from pairwiseprediction.optimized import OptimizedPairwiseClassifier
from scipy.stats import poisson, uniform
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from xgboost import XGBClassifier as XGBc

from germina.tree import DTR

warnings.filterwarnings("ignore")

__ = enable_iterative_imputer


def imputer(alg, n_estimators, seed, jobs):
    if alg == "lgbm":
        return IterativeImputer(LGBMr(n_estimators=n_estimators, random_state=seed, n_jobs=jobs, deterministic=True, force_row_wise=True), random_state=seed)
    elif alg.endswith("knn"):
        return IterativeImputer(Pipeline(steps=[("scaler", StandardScaler()), ("knn", KNeighborsRegressor(n_neighbors=n_estimators, n_jobs=jobs))]), random_state=seed)
    else:
        raise Exception(f"Unknown {alg=}")


def imputation(Xy_tr, babya, babyb, alg_imp, n_estimators_imp, seed, jobs):
    print("\timputing", end="", flush=True)
    imp = imputer(alg_imp, n_estimators_imp, seed, jobs)
    Xy_tr = imp.fit_transform(Xy_tr)  # First, fit using label info.
    imp.fit(Xy_tr[:, :-1])  # Then fit without labels to be able to make a model compatible with the test instance.
    babyxa = imp.transform(babya[:, :-1])
    babyxb = imp.transform(babyb[:, :-1])
    babya[:, :-1] = babyxa
    babyb[:, :-1] = babyxb
    return Xy_tr, babya, babyb


def predictors(alg, n_estimators, seed, jobs):
    match alg:
        case "rf":
            return RFc, {"n_estimators": n_estimators, "random_state": seed, "n_jobs": jobs}
        case "lgbm":
            return LGBMc, {"deterministic": True, "force_row_wise": True, "random_state": seed, "n_jobs": jobs}
        case "et":
            return ETc, {"n_estimators": n_estimators, "random_state": seed, "n_jobs": jobs}
        case "xg":
            return XGBc, {"n_estimators": n_estimators, "random_state": seed, "n_jobs": jobs}
        case "cart":
            return DecisionTreeClassifier, {"random_state": seed}
        case x:
            if x.startswith("cart"):
                param_dist = {
                    "criterion": ["gini", "entropy"],
                    "max_depth": poisson(mu=5, loc=2),
                    "min_impurity_decrease": uniform(0, 0.01),
                    "max_leaf_nodes": poisson(mu=20, loc=5),
                    "min_samples_split": ap[20, 30, ..., 100].l,
                    "min_samples_leaf": ap[10, 20, ..., 50].l,
                    "random_state": seed,
                }
                return DecisionTreeClassifier, param_dist
            elif x.startswith("ocart"):
                param_dist = {
                    "criterion": ["gini", "entropy"],
                    "max_depth": poisson(mu=5, loc=2),
                    "min_impurity_decrease": uniform(0, 0.01),
                    "max_leaf_nodes": poisson(mu=20, loc=5),
                    "min_samples_split": ap[20, 30, ..., 100].l,
                    "min_samples_leaf": ap[10, 20, ..., 50].l,
                }
                n_iter = int(x.split("-")[1])
                cv = int(x.split("-")[2])
                clf = DecisionTreeClassifier()
                return RandomizedSearchCV, {
                    "pre_dispatch": "n_jobs//2",
                    "cv": cv,
                    "n_jobs": jobs,
                    "estimator": clf,
                    "param_distributions": param_dist,
                    "n_iter": n_iter,
                    "random_state": seed,
                    "scoring": "balanced_accuracy",
                }
            else:
                raise Exception(f"Unknown {alg=}. Options: rf,lgbm,et,xg,cart,ocart-*")


def trainpredict_c(Xwtr, Xwts,
                   alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
                   n_estimators_train, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    predictors_, kwargs = predictors(alg_train, n_estimators_train, seed, jobs)
    alg_c = PairwiseClassifier(predictors_,
                               pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(Xwtr)
    predicted_c = alg_c.predict(Xwts, paired_rows=True)[::2]
    predictedprobas_c = alg_c.predict_proba(Xwts, paired_rows=True)[::2]
    return predicted_c, predictedprobas_c


def trainpredictshap(Xwtr, Xwts,
                     alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
                     n_estimators_train, columns, seed, jobs):
    print("\ttrainingC", end="", flush=True)
    predictors_, kwargs = predictors(alg_train, n_estimators_train, seed, jobs)
    alg_c = PairwiseClassifier(predictors_,
                               pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(Xwtr)
    print("\tpredictingC", end="", flush=True)
    predicted_labels = alg_c.predict(Xwts, paired_rows=True)[::2]
    predicted_probas = alg_c.predict_proba(Xwts, paired_rows=True)[::2]
    print("\tcalculating SHAP", end="", flush=True)
    shap = alg_c.shap(Xwts[0], Xwts[1], columns, seed)
    return predicted_labels, predicted_probas, shap


def shap_for_pair(best_params, xa, xb, Xw, alg_train, n_estimators_train, pairing_style, proportion, threshold, columns, seed, jobs, **kwargs):
    """
    Return an indexed dict {variable: (value, SHAP)} for the given pair of instances

    >>> import numpy as np
    >>> from sklearn.datasets import load_diabetes
    >>> from sklearn.ensemble import RandomForestClassifier
    >>> a, b = load_diabetes(return_X_y=True)
    >>> me = np.mean(b)
    >>> # noinspection PyUnresolvedReferences
    >>> y = (b > me).astype(int)
    >>> c = b.reshape(len(b), 1)
    >>> Xw = np.hstack([a, c])
    >>> columns = [str(i) for i in range(a.shape[1])]
    >>> params = {'criterion': 'gini', 'max_depth': 7, 'max_leaf_nodes': 31, 'min_impurity_decrease': 0.00652999760945243, 'min_samples_leaf': 40, 'min_samples_split': 40}
    >>> shap_for_pair("cart-2-2", params, 0, Xw[0,:], Xw[1,:], Xw, "difference", False, 0.2, columns, 0, -1)
    """
    import dalex as dx
    from indexed import Dict
    from pandas import DataFrame

    # prepare Xtr,ytr and pair
    handle_last_as_y = "%" if proportion else True
    filter = lambda tmp: (tmp[:, -1] < -threshold) | (tmp[:, -1] >= threshold)
    if pairing_style == "difference":
        x = pairwise_diff(xa[:, :-1], xb[:, :-1])
        pairs = lambda a, b: pairwise_diff(a, b, pct=handle_last_as_y == "%")
    elif pairing_style == "concatenation":
        x = pairwise_hstack(xa[:, :-1], xb[:, :-1])
        pairs = lambda a, b: pairwise_hstack(a, b, handle_last_as_y=handle_last_as_y)
    else:
        raise Exception(f"Not implemented for {pairing_style=}")
    idxs = np.argsort(Xw[:, -1].flatten(), kind="stable").flatten()
    Xw = Xw[idxs]
    tmp = pairs(Xw, Xw)
    pairs_Xy_tr = tmp[filter(tmp)]
    Xtr = pairs_Xy_tr[:, :-1]
    ytr = (pairs_Xy_tr[:, -1] >= 0).astype(int)
    if pairing_style == "concatenation":
        f = lambda i: [f"{i}_{col}" for col in columns]
        columns = (f("a") + f("b"))
    x = DataFrame(x, columns=columns)
    Xtr = DataFrame(Xtr, columns=columns)

    # fit
    print("\tcalculating SHAP", end="", flush=True)
    alg, kwargs_ = predictors(alg_train, n_estimators_train, seed, jobs)
    estimator = alg(**best_params)
    estimator.fit(Xtr, ytr)

    # SHAP
    explainer = dx.Explainer(model=estimator, data=Xtr, y=ytr, verbose=False)
    predictparts = dx.Explainer.predict_parts(explainer, new_observation=x, type="shap", random_state=seed, **kwargs)
    zz = zip(predictparts.result["variable"], predictparts.result["contribution"])
    var__val_shap = Dict((name_val.split(" = ")[0], (float(name_val.split(" = ")[1:][0]), co)) for name_val, co in zz)
    return var__val_shap


def trainpredict_optimized(Xwtr, Xwts,
                           tries, kfolds,
                           alg_train, pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction,
                           n_estimators_train, seed, jobs):
    print("\toptimizing", end="", flush=True)
    predictor, kwargs = predictors(alg_train, n_estimators_train, seed, jobs)
    kwargs_ = {"random_state": kwargs.pop("random_state")} if "random_state" in kwargs else {}
    algo = OptimizedPairwiseClassifier(kwargs, tries, kfolds, seed, predictor,
                                       pairing_style, threshold, proportion, center, only_relevant_pairs_on_prediction, **kwargs_)
    algo.fit(Xwtr)
    pred = algo.predict(Xwts, paired_rows=True)[::2]
    prob = algo.predict_proba(Xwts, paired_rows=True)[::2]
    return {"pred": pred, "prob": prob, "best_params": algo.best_params, "best_score": algo.best_score, "opt_results": algo.opt_results}


def get_algclass(name):
    if name.startswith("cartr"):
        return DecisionTreeRegressor
    elif name.startswith("dtr"):
        return DTR
    elif name.startswith("rfr"):
        return RFr
    raise Exception(f"{name=}")


def get_algspace(name):
    if name.startswith("cartr"):  # no "poisson" due to negative values in diff of pairs
        return {"criterion": ["absolute_error", "squared_error", "friedman_mse"], "max_depth": poisson(mu=5, loc=2), "min_impurity_decrease": uniform(0, 0.01), "max_leaf_nodes": poisson(mu=20, loc=5), "min_samples_split": ap[20, 30, ..., 100].l, "min_samples_leaf": ap[10, 20, ..., 50].l}
    elif name.startswith("dtr"):
        return {"max_depth": poisson(mu=5, loc=2)}
    elif name.startswith("rfr100"):
        return {'n_estimators': 100, "n_jobs": -1}
    elif name.startswith("rfr1000"):
        return {'n_estimators': 1000, "n_jobs": -1}
    elif name.startswith("rfr5000"):
        return {'n_estimators': 5000, "n_jobs": -1}
    elif name.startswith("rfro10"):
        return {"n_estimators": [10],"criterion": ["absolute_error", "squared_error", "friedman_mse"], "max_depth": poisson(mu=5, loc=2), "min_impurity_decrease": uniform(0, 0.01), "max_leaf_nodes": poisson(mu=20, loc=5), "min_samples_split": ap[20, 30, ..., 100].l, "min_samples_leaf": ap[10, 20, ..., 50].l}
    elif name.startswith("rfro100"):
        return {"n_estimators": [100],"criterion": ["absolute_error", "squared_error", "friedman_mse"], "max_depth": poisson(mu=5, loc=2), "min_impurity_decrease": uniform(0, 0.01), "max_leaf_nodes": poisson(mu=20, loc=5), "min_samples_split": ap[20, 30, ..., 100].l, "min_samples_leaf": ap[10, 20, ..., 50].l}
    raise Exception(f"{name=}")


def fit(algname, params, df, verbose=True):
    if verbose:
        print(f"\tfitting {algname.split('-')[0]}", end="", flush=True)
    estimator = get_algclass(algname)(**params)
    estimator.fit(df.iloc[:, :-1], df.iloc[:, -1])
    return estimator


def fitpredict(algname, params, Xwtr, Xts, verbose=True):
    if verbose:
        print("\tfitpredicting", end="", flush=True)
    estimator = fit(algname, params, Xwtr, verbose=False)
    zts = estimator.predict(Xts)
    return zts


def fitshap(algname, params, Xy, seed, njobs, verbose=True, **kwargs):
    def job(idx):
        Xwtr = Xy.drop([idx], axis="rows")
        ytr = Xwtr.iloc[:, -1]
        xy = Xy.loc[idx]
        x = xy[:-1]
        estimator = fit(algname, params, Xwtr, verbose=False)
        explainer = dx.Explainer(model=estimator, data=Xwtr.iloc[:, :-1], y=ytr, verbose=False)
        predictparts = dx.Explainer.predict_parts(explainer, new_observation=x, type="shap", random_state=seed, **kwargs)
        zz = zip(predictparts.result["variable"], predictparts.result["contribution"])
        var__val_shap = Dict((name_val.split(" = ")[0], (float(name_val.split(" = ")[1:][0]), co)) for name_val, co in zz)
        return var__val_shap

    if verbose:
        print("\tfitshap", end="", flush=True)
    lst = []
    for dct in Parallel(n_jobs=njobs)(delayed(job)(idx) for idx in Xy.index):
        lst.append(dct)
    return lst


def fitshap2(algname, params, Xy, xa, xb, seed, verbose=True, **kwargs):
    def job(x):
        predictparts = dx.Explainer.predict_parts(explainer, new_observation=x, type="shap", random_state=seed, **kwargs)
        zz = zip(predictparts.result["variable"], predictparts.result["contribution"])
        return Dict((name_val.split(" = ")[0], (float(name_val.split(" = ")[1:][0]), co)) for name_val, co in zz)

    if verbose:
        print("\tfitshap2", end="", flush=True)
    X = Xy.iloc[:, :-1]
    y = Xy.iloc[:, -1]
    estimator = fit(algname, params, Xy, verbose=False)
    explainer = dx.Explainer(model=estimator, data=X, y=y, verbose=False)
    lst = []
    for var__val_shap in Parallel(n_jobs=2)(delayed(job)(x) for x in [xa, xb]):
        lst.append(var__val_shap)
    return lst
