import warnings
from datetime import datetime
from itertools import repeat

from shelchemy.scheduler import Scheduler

warnings.filterwarnings('ignore')
from pprint import pprint
from sys import argv

from argvsucks import handle_command_line
from germina.runner import ch
from hdict import hdict, apply, _
from shelchemy import sopen
from sklearn.experimental import enable_iterative_imputer

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.dataset import join
from germina.loader import load_from_csv, clean_for_dalex, get_balance, train_xgb, build_explainer, explain_modelparts, explain_predictparts

from sklearn.model_selection import LeaveOneOut

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
with (sopen(local_cache_uri) as local_storage, sopen(near_cache_uri) as near_storage, sopen(remote_cache_uri) as remote_storage, sopen(schedule_uri) as db):
    storages = {
        "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }

    for arq, field, oldidx in [("t_3-4_pathways_filtered", "pathways34", "Pathways"),
                               ("t_3-4_species_filtered", "species34", "Species"),
                               ("t_5-7_pathways_filtered", "pathways57", "Pathways"),
                               ("t_5-7_species_filtered", "species57", "Species"),
                               ("t_8-9_pathways_filtered", "pathways89", "Pathways"),
                               ("t_8-9_species_filtered", "species89", "Species")]:
        print(field, "=================================================================================")
        d = load_from_csv(d, storages, storage_to_be_updated, path, vif, arq, field, transpose=True, old_indexname=oldidx)
        d = load_from_csv(d, storages, storage_to_be_updated, path, False, "target_EBF", "ebf", False)

        d = d >> apply(join, df=_.ebf, other=_[field]).df
        d = ch(d, storages, storage_to_be_updated)

        d = clean_for_dalex(d, storages, storage_to_be_updated)
        d = get_balance(d, storages, storage_to_be_updated)

        params = {"max_depth": 5, "objective": "binary:logistic", "eval_metric": "auc"}
        loo = LeaveOneOut()
        runs = loo.split(d.X)
        tasks = zip(repeat(field), range(len(d.X)))
        for f, i in (Scheduler(db, timeout=60) << tasks) if sched else tasks:
            idxtr, idxts = next(runs)
            print(f"\tts:{idxts}\t", datetime.now(), f"\t{100 * i / len(d.X):1.1f} %\t-----------------------------------")
            d = d >> apply(train_xgb, params, idxtr=idxtr).classifier
            d = ch(d, storages, storage_to_be_updated)

            d = d >> apply(build_explainer, idxtr=idxtr).explainer
            d = ch(d, storages, storage_to_be_updated)

            d = d >> apply(explain_modelparts).modelparts
            d = ch(d, storages, storage_to_be_updated)

            d = d >> apply(explain_predictparts, idxts=idxts).predictparts
            d = ch(d, storages, storage_to_be_updated)

            # modelparts: VariableImportance = d.modelparts
            # pprint(modelparts.result[["variable", "contribution"]].to_dict())

            # predictparts: VariableImportance = d.predictparts
            # varcontrib = dict(list(sorted(zip(predictparts.result["contribution"], predictparts.result["variable"]), key=lambda x: x[0]))[:5])
            # pprint(varcontrib)

            # d.modelparts.plot(show=False).show()
            # d.predictparts.plot(min_max=[0, 1], show=False).show()

        print()
