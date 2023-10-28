from datetime import datetime

from hdict import apply, _
from hdict.dataset.pandas_handling import file2df
from pandas import DataFrame
from sklearn.preprocessing import StandardScaler

from germina.runner import drop_many_by_vif, ch, sgid2estudoid, setindex


def load_from_csv(d, storages, storage_to_be_updated, path, vif, filename, field, transpose, old_indexname="id_estudo"):
    print(datetime.now())
    d = d >> apply(file2df, path + filename + ".csv", transpose=transpose, index=True)(field)
    d = ch(d, storages, storage_to_be_updated)
    print(f"Loaded '{field}' data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    if vif:
        print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], [])
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    print()
    d = d >> apply(setindex, _[field], old_indexname=old_indexname)(field)
    d = ch(d, storages, storage_to_be_updated)
    return d


def load_from_synapse(d, storages, storage_to_be_updated, path, vif, filename, field):
    print(datetime.now())
    d = d >> apply(file2df, path + filename + ".csv")(field)
    d = ch(d, storages, storage_to_be_updated)
    print(f"Loaded '{field}' Synapse data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = d >> apply(sgid2estudoid, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    print("Fixed id ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓\n", d[field], "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    if vif:
        print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], [])
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    print()
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    return d


def load_from_osf(d, storages, storage_to_be_updated, path, vif, filename, vars__no_t, field, keeprows):
    print(datetime.now())
    if field == "fullosf":
        d = d >> apply(file2df, path + filename + ".csv")(field)
    else:
        if "fullosf" not in d:
            raise Exception(f"Load 'fullosf' from csv first.")
        vars = []
        for v in sorted(vars__no_t):
            for i in range(7):
                t_candidate = f"{v}_t{i}"
                if t_candidate in d.fullosf:
                    vars.append(t_candidate)
        vars.sort()
        d = d >> apply(lambda fullosf, vs: fullosf[vs], vs=vars)(field)
    if vif:
        print(f"Apply VIF to '{field}' ----------------------------------------------------------------------------------------------------------------------------")
        print(datetime.now())
        d = drop_many_by_vif(d, field, storages, storage_to_be_updated, [], keeprows)
        d = ch(d, storages, storage_to_be_updated)
        print("after VIF ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑")
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    print(f"Loaded '{field}' OSF data from '{filename}.csv' ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d


def apply_std(d, storages, storage_to_be_updated, path, vif, field):
    print(datetime.now())
    d = d >> apply(lambda x: DataFrame(StandardScaler().fit_transform(x)), _[field])(field)
    d = d >> apply(setindex, _[field])(field)
    d = ch(d, storages, storage_to_be_updated)
    print("Scaled ↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓↓", d[field].shape, "↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑↑\n")
    return d
