from sys import argv

import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from lange import ap
from pandas import read_csv
from scipy.stats import ttest_1samp
from shelchemy import sopen

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri
from germina.loo import loo

dct = handle_command_line(argv, delta=float, trees=int, pct=False, demo=False, sched=False, perms=1, targetvar=str, jobs=int, alg=str, seed=0, prefix=str, sufix=str, trees_imp=int, feats=int, tfsel=int, forward=False, pairwise=str)
print(dct)
trees, delta, pct, demo, sched, perms, targetvar, jobs, alg, seed, prefix, sufix, trees_imp, feats, tfsel, forward, pairwise = dct["trees"], dct["delta"], dct["pct"], dct["demo"], dct["sched"], dct["perms"], dct["targetvar"], dct["jobs"], dct["alg"], dct["seed"], dct["prefix"], dct["sufix"], dct["trees_imp"], dct["feats"], dct["tfsel"], dct["forward"], dct["pairwise"]
rnd = np.random.default_rng(0)
handle_last_as_y = "%" if pct else True
with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        # "remote": remote_storage,
        "near": near_storage,
        "local": local_storage,
    }
    for sp in [1, 2]:
        print(f"{sp=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        df = read_csv(f"{prefix}{sp}{sufix}", index_col="id_estudo")
        df.sort_values(targetvar, inplace=True, ascending=True, kind="stable")
        if demo:
            take = min(df.shape[0] // 2, 30)
            df = pd.concat([df.iloc[:take], df.iloc[-take:]], axis="rows")
        print(df.shape, "<<<<<<<<<<<<<<<<<")
        age = df["idade_crianca_dias_t2"]
        ret = loo(df, permutation=0, pairwise=pairwise, threshold=delta, rejection_threshold=0,
                  alg=alg, n_estimators=trees,
                  n_estimators_imp=trees_imp,
                  n_estimators_fsel=tfsel, forward_fsel=forward, k_features_fsel=feats, k_folds_fsel=4,
                  db=db, storages=storages, sched=sched,
                  seed=seed, jobs=jobs)
        if ret:
            d, bacc_c0, bacc_r0, r2_c0, r2_r0, hits_c0, hits_r0, tot0, tot_c0, tot_r0, rj_c0, rj_r0 = ret
            print(f"\r{sp=} {delta=} {trees=} {bacc_c0=:4.3f} {bacc_r0=:4.3f} | {r2_c0=:4.3f} {r2_r0=:4.3f} {hits_c0=}  {hits_r0=} {tot0=} {tot_c0=} {tot_r0=}\t{d.hosh.ansi} | {rj_c0=} {rj_r0=}", flush=True)

        # permutation test
        scores_dct = dict(bacc_c=[], bacc_r=[], r2_c=[], r2_r=[])
        for permutation in ap[1, 2, ..., perms]:
            df_shuffled = df.copy()
            df_shuffled[targetvar] = rnd.permutation(df[targetvar].values)
            ret = loo(df_shuffled, permutation, pairwise=pairwise, threshold=delta, rejection_threshold=0,
                      alg=alg, n_estimators=trees,
                      n_estimators_fsel_imput=tfsel, forward_fsel=forward, k_features_fsel=feats, k_folds_fsel=4,
                      db=db, storages=storages, sched=sched,
                      seed=seed, jobs=jobs)
            if ret:
                d, bacc_c, bacc_r, r2_c, r2_r, hits_c, hits_r, tot, tot_c, tot_r, rj_c, rj_r = ret
                scores_dct["bacc_c"].append(bacc_c - bacc_c0)
                scores_dct["bacc_r"].append(bacc_r - bacc_r0)
                scores_dct["r2_c"].append(r2_c - r2_c0)
                scores_dct["r2_r"].append(r2_r - r2_r0)

        if sched:
            print("Run again without providing flag `sched`.")
            continue

        if scores_dct:
            print(f"\n{sp=} p-values: ", end="")
            for measure, scores in scores_dct.items():
                p = ttest_1samp(scores, popmean=0, alternative="greater")[1]
                print(f"  {measure}={p:4.3f}", end="")
        print("\n")

"""s.o.s
# filtered species t1|t2
j=-1;s="sched";t=32;a=lgbm;pre="results/datasetr_species";suf="_bayley_8_t2.csv"; ps;     for p in $(seq 0 9999); do  time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct diff $s; done # filtered species
p=0;j=1;s="";t=32;a=lgbm;pre="results/datasetr_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct diff $s; # filtered species

# full species t1|t2 7.5
j=-1;s="sched";t=1024;a=lgbm;pre="results/datasetr_fromtsv_species";suf="_bayley_8_t2.csv"; ps;     for p in $(seq 0 9999); do  time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p $s; done # full species
p=0;j=1;s="";t=1024;a=lgbm;pre="results/datasetr_fromtsv_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p $s; # full species
# full species t1|t2 pct
j=-1;s="sched";t=1024;a=lgbm;pre="results/datasetr_fromtsv_species";suf="_bayley_8_t2.csv"; ps;     for p in $(seq 0 9999); do  time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct diff $s; done # full species
p=0;j=1;s="";t=1024;a=lgbm;pre="results/datasetr_fromtsv_species";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct diff $s; # full species

# eeg single|dyadic 7.5
j=-1;s="sched";t=1024;a=lgbm;pre="results/single_or_dyadic_is";suf="_bayley_8_t2.csv"; ps;     for p in $(seq 0 9999); do  time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p $s; done # eeg
p=0;j=1;s="";t=1024;a=lgbm;pre="results/single_or_dyadic_is";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p diff $s; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=7.5 trees=$t jobs=$j perms=$p $s; # eeg
# eeg single|dyadic  pct
j=-1;s="sched";t=1024;a=lgbm;pre="results/single_or_dyadic_is";suf="_bayley_8_t2.csv"; ps;     for p in $(seq 0 9999); do  time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct diff $s; done # eeg
p=0;j=1;s="";t=1024;a=lgbm;pre="results/single_or_dyadic_is";suf="_bayley_8_t2.csv"; ps; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct $s; time poetry run python experiments/microbiome/pairwise_v2.py trees_imp=10 prefix=$pre sufix=$suf targetvar=bayley_8_t2 alg=$a delta=0.2 trees=$t jobs=$j perms=$p pct diff $s; # eeg
"""
