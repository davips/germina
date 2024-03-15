from itertools import islice

import numpy as np
from joblib.parallel import Parallel, delayed
from pairwiseprediction.classifier import PairwiseClassifier
from pairwiseprediction.optimized import OptimizedPairwiseClassifier
from sklearn.metrics import r2_score, balanced_accuracy_score
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, ParameterSampler

from germina.aux import predictors, get_algclass, get_algspace, fitpredict
from germina.sampling import pairwise_sample


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
    return alg_c._estimator, alg_c.best_params, alg_c.best_score, alg_c.opt_results


def tree_optimized(df, alg: RandomizedSearchCV, verbose=False):
    if verbose:
        print("\tOptimizing reg. tree hyperparameters.", end="", flush=True)
    X = df.iloc[:, :-1]
    y = df.iloc[:, -1]
    alg.fit(X, y)
    return alg.best_estimator_, alg.best_params_, alg.best_score_, alg.cv_results_


# TODO: stratification is wrong as w is always positive
def tree_optimized_dv(df, n_iter, start, end, k, algname, seed=0, njobs=16, verbose=False):
    if verbose:
        print("\tOptimizing reg. tree hyperparameters.", end="", flush=True)
    Xw = df.to_numpy()
    w = df.iloc[:, -1].to_numpy()
    algclass = get_algclass(algname)
    search_space = get_algspace(algname)

    # noinspection PyUnresolvedReferences
    y = (w >= 0).astype(int)  # for "class stratification" of the regression value across folds
    skf = StratifiedKFold(n_splits=k, random_state=seed, shuffle=True)

    def job(params_):
        ytss, ztss = [], []
        for train_index, test_index in skf.split(Xw, y):
            # prepare data sets
            Xtr = Xw[train_index, :-1]
            ytr = Xw[train_index, -1]
            Xts = Xw[test_index, :-1]
            yts = Xw[test_index, -1]

            # train/predict with sampled arguments
            regr = algclass(**params_)
            regr.fit(Xtr, ytr)
            zts = regr.predict(Xts)

            # accumulate results
            ytss.extend(yts)
            ztss.extend(zts)
        score = r2_score(ztss, ytss)
        return score, params_

    sampler = islice(ParameterSampler(search_space, n_iter, random_state=seed), start, end)
    best_score = -1000
    for score, params in Parallel(n_jobs=njobs)(delayed(job)(params) for params in sampler):
        if score > best_score:
            best_score = score
            best_params = params
    return best_params.copy(), best_score


def tree_optimized_dv_pair(df, trials, start, end, algname, seed=0, njobs=16, verbose=False):
    """

    :param df:
    :param trials:  number of pairs
    :param start:
    :param end:
    :param algname:
    :param seed:
    :param njobs:
    :param verbose:
    :return:
    """
    if verbose:
        print("\tOptimizing reg by pair.", end="", flush=True)
    search_space = get_algspace(algname)
    pairs = pairwise_sample(df.index.tolist(), trials, seed)

    def job(params_):
        vts_lst, wts_lst = [], []  # continuous
        yts_lst, zts_lst = [], []  # binary
        for idxa, idxb in pairs:
            # prepare current pair of babies for testing and build training set
            babydfa = df.loc[[idxa], :]
            babydfb = df.loc[[idxb], :]
            baby_va = babydfa.iloc[0, -1:].to_numpy()
            baby_vb = babydfb.iloc[0, -1:].to_numpy()
            babya = babydfa.to_numpy()
            babyb = babydfb.to_numpy()
            Xv_tr = df.drop([idxa, idxb], axis="rows")
            Xv_ts = np.vstack([babya, babyb])

            # train/predict with sampled arguments
            wts = fitpredict(algname, params_, Xv_tr, Xv_ts[:, :-1], verbose=False)

            # accumulate
            vts_lst.extend([baby_va[0], baby_vb[0]])
            wts_lst.extend([wts[0], wts[1]])
            yts_lst.append(baby_va[0] >= baby_vb[0])
            zts_lst.append(wts[0] >= wts[1])
        r2_ = r2_score(vts_lst, wts_lst)
        bacc_ = balanced_accuracy_score(yts_lst, zts_lst)
        return r2_, bacc_, params_

    sampler = islice(ParameterSampler(search_space, trials, random_state=seed), start, end)
    best_r2 = best_bacc = -1000
    best_r2_params = best_bacc_params = None
    for r2, bacc, params in Parallel(n_jobs=njobs)(delayed(job)(params) for params in sampler):
        if best_r2 > r2:
            best_r2 = r2
            best_r2_params = params
        if best_bacc > bacc:
            best_bacc = bacc
            best_bacc_params = params
    return best_r2_params.copy(), best_bacc_params.copy(), best_r2, best_bacc
