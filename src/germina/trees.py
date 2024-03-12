import numpy as np
from pairwiseprediction.classifier import PairwiseClassifier
from pairwiseprediction.optimized import OptimizedPairwiseClassifier
from sklearn.model_selection import RandomizedSearchCV

from germina.l2o import predictors


def pwtree(df, alg, seed, jobs, pairwise, delta, proportion=False, center=None, only_relevant_pairs_on_prediction=False, verbose=False):
    if verbose:
        print("Optimizing tree hyperparameters.")
    Opt: RandomizedSearchCV
    Opt, kwargs = predictors(alg, None, seed, jobs)
    alg_c = PairwiseClassifier(Opt, pairwise, delta, proportion=proportion, center=center, only_relevant_pairs_on_prediction=only_relevant_pairs_on_prediction, **kwargs)
    alg_c.fit(df)
    opt = alg_c._estimator
    return opt.best_estimator_, opt.best_params_, opt.best_score_, opt.cv_results_


def pwtree_optimized(df, alg, tries, kfolds, seed, jobs, pairwise, delta, proportion=False, center=None, only_relevant_pairs_on_prediction=False, verbose=False):
    if verbose:
        print("Optimizing tree hyperparameters.")
    predictor, kwargs = predictors(alg, None, seed, jobs)
    kwargs_ = {"random_state": kwargs.pop("random_state")} if "random_state" in kwargs else {}
    alg_c = OptimizedPairwiseClassifier(kwargs, tries, kfolds, seed, predictor, pairwise, delta, proportion=proportion, center=center, only_relevant_pairs_on_prediction=only_relevant_pairs_on_prediction, **kwargs_)
    alg_c.fit(df)
    return alg_c._estimator, alg_c.best_params, alg_c.best_score


def report(results, n_top=3):
    for i in range(1, n_top + 1):
        candidates = np.flatnonzero(results["rank_test_score"] == i)
        for candidate in candidates:
            print("Model with rank: {0}".format(i))
            print(
                "Mean validation score: {0:.3f} (std: {1:.3f})".format(
                    results["mean_test_score"][candidate],
                    results["std_test_score"][candidate],
                )
            )
            print("Parameters: {0}".format(results["params"][candidate]))
            print("")
