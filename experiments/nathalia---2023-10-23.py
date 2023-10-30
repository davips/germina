from datetime import datetime
import dalex as dx

import pandas as pd
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

import xgboost as xgb

import warnings

warnings.filterwarnings('ignore')
from pprint import pprint
from sys import argv

import numpy as np
from argvsucks import handle_command_line
from germina.runner import ch
from hdict import hdict, apply, _
from pandas import DataFrame
from shelchemy import sopen
from sklearn.experimental import enable_iterative_imputer

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri
from germina.dataset import osf_except_target_vars__no_t, eeg_vars__no_t, join
from germina.loader import load_from_osf, load_from_synapse, load_from_csv, clean_for_dalex, get_balance, train_xgb, build_explainer, explain_modelparts, explain_predictparts
import dalex as dx

import numpy as np
import pandas as pd

from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.model_selection import RandomizedSearchCV

import warnings

warnings.filterwarnings('ignore')
__ = enable_iterative_imputer
dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vif=False, nans=False, sched=False, up="")
print(datetime.now())
pprint(dct, sort_dicts=False)
print()
path = "data/paper-breastfeeding/"
d = hdict(
    n_permutations=dct["pvalruns"],
    n_repeats=dct["importanceruns"],
    imputrees=dct["imputertrees"],
    random_state=dct["seed"],
    target_var=dct["target"],  # "ibq_reg_cat_t3", bayley_average_t4
    max_iter=dct["trees"], n_estimators=dct["trees"],
    n_splits=5,
    index="id_estudo", join="inner", shuffle=True, n_jobs=-1, return_name=False,
    osf_filename="germina-osf-request---davi121023"
)
vif, nans, sched, storage_to_be_updated = dct["vif"], dct["nans"], dct["sched"], dct["up"]

with (sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    d = load_from_csv(d, storages, storage_to_be_updated, path, vif, "t_3-4_pathways_filtered", "pathways34", transpose=True, old_indexname="Pathways")
    d = load_from_csv(d, storages, storage_to_be_updated, path, vif, "t_3-4_species_filtered", "species34", transpose=True, old_indexname="Species")
    d = load_from_csv(d, storages, storage_to_be_updated, path, False, "target_EBF", "ebf", False)

    d = d >> apply(join, df=_.ebf, other=_.pathways34).df
    d = ch(d, storages, storage_to_be_updated)

    d = clean_for_dalex(d, storages, storage_to_be_updated)
    d = get_balance(d, storages, storage_to_be_updated)
# n = 500
# m = 100
# X, y = d.X.iloc[:n, :m], d.y.iloc[:n]
X, y = d.X, d.y

####################################################################

params = {
    "max_depth": 5,
    "objective": "binary:logistic",
    "eval_metric": "auc"
}

loo = LeaveOneOut()
for i, (idxtr, idxts) in enumerate(loo.split(X)):
    d = d >> apply(train_xgb, params, idxtr=idxtr).classifier
    d = ch(d, storages, storage_to_be_updated)

    d = d >> apply(build_explainer, idxtr=idxtr).explainer
    d = ch(d, storages, storage_to_be_updated)

    d = d >> apply(explain_modelparts).modelparts
    d = ch(d, storages, storage_to_be_updated)

    d = d >> apply(explain_predictparts, idxts=idxts).predictparts
    d = ch(d, storages, storage_to_be_updated)
