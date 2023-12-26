from datetime import datetime
from pprint import pprint
from sys import argv

import numpy as np
from argvsucks import handle_command_line
from hdict import hdict, apply, _
from pandas import DataFrame
from shelchemy import sopen
from sklearn.experimental import enable_iterative_imputer

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri
from germina.dataset import osf_except_target_vars__no_t, eeg_vars__no_t, join
from germina.loader import load_from_osf, load_from_synapse

__ = enable_iterative_imputer
dct = handle_command_line(argv, pvalruns=int, importanceruns=int, imputertrees=int, seed=int, target=str, trees=int, vif=False, nans=False, sched=False, up="")
print(datetime.now())
pprint(dct, sort_dicts=False)
print()
path = "data/"
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

    d = load_from_osf(d, storages, storage_to_be_updated, path, False, d.osf_filename, osf_except_target_vars__no_t, "fullosf", [])
    d = load_from_synapse(d, storages, storage_to_be_updated, path, False, "data/eeg-synapse-reduced.csv", "eegsyn")
    d = load_from_osf(d, storages, storage_to_be_updated, path, vif, d.osf_filename, eeg_vars__no_t, "eegosf", keeprows=d.eegsyn.index)

    d = d >> apply(lambda _: join(DataFrame({"id_estudo": _.eegsyn.index}), index="id_estudo", other=_.eegosf, join="inner")).eegosf_selected
    d = d >> apply(lambda _: join(DataFrame({"id_estudo": list(set(_.eegosf.index).difference(_.eegsyn.index))}), index="id_estudo", other=_.eegosf, join="inner")).eegosf_left

    print(d.fullosf)
    print(d.fullosf.shape)
    print(d.eegsyn.shape)
    print(d.eegosf_left)
    pprint(d.eegosf.to_dict()["Beta_t1"])
    pprint(d.eegosf_left.to_dict()["Beta_t1"])
    # print(d.eegosf_selected)

    # print(d.eegosf)
    # df:DataFrame=d.eegosf
    # df.to_csv("/tmp/eegosf.csv")

    print(d.eegosf_selected)
    df:DataFrame=d.eegosf_selected
    df.to_csv("/tmp/eegosf_selected.csv")
