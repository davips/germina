from sys import argv

import numpy as np
import pandas as pd
from argvsucks import handle_command_line
from lange import ap
from pandas import read_csv
from scipy.stats import ttest_1samp
from shelchemy import sopen

from germina.config import local_cache_uri, remote_cache_uri, near_cache_uri, schedule_uri

dct = handle_command_line(argv, noage=False, delta=float, trees=int, pct=False, demo=False, sched=False, perms=1, jobs=int, alg=str, seed=0, trees_imp=int, feats=int, tfsel=int, forward=False, pairwise=str, sps=list)
print(dct)
noage, trees, delta, pct, demo, sched, perms, jobs, alg, seed, trees_imp, feats, tfsel, forward, pairwise, sps = dct["noage"], dct["trees"], dct["delta"], dct["pct"], dct["demo"], dct["sched"], dct["perms"], dct["jobs"], dct["alg"], dct["seed"], dct["trees_imp"], dct["feats"], dct["tfsel"], dct["forward"], dct["pairwise"], dct["sps"]
rnd = np.random.default_rng(0)
with (sopen(local_cache_uri, ondup="skip") as local_storage, sopen(near_cache_uri, ondup="skip") as near_storage, sopen(remote_cache_uri, ondup="skip") as remote_storage, sopen(schedule_uri) as db):
    storages = {
        "remote": remote_storage,
        # "near": near_storage,
        "local": local_storage,
    }
    for sp in sps:
        sp = int(sp)
        print(f"{sp=} <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
        mother_df = read_csv(f"data/nathalia060224-mother-imc.csv", index_col="id_estudo")
        targets_df = read_csv(f"data/nathalia260224-targets.csv", index_col="id_estudo")
        species_df = read_csv(f"data/full/T{sp}_especies_original.csv", index_col="id_estudo")
        X_df = species_df.join(mother_df, how="inner")
        alltargets = ['bayley_3_t1', 'bayley_3_t2', 'bayley_3_t3', 'bayley_8_t1',
                      'bayley_8_t2', 'bayley_8_t3', 'ibq_reg_t1', 'ibq_reg_t2', 'ibq_reg_t3',
                      'ibq_soot_t1', 'ibq_soot_t2', 'ibq_soot_t3', 'ibq_dura_t1',
                      'ibq_dura_t2', 'ibq_dura_t3', 'bayley_3_t4', 'bayley_8_t4',
                      'ecbq_atf_t4', 'ecbq_ats_t4', 'ecbq_inh_t4', 'ecbq_sth_t4',
                      'ecbq_effco_t4']
        targets = [tgt for tgt in alltargets if tgt[-1] == "2"]
        Y_df = targets_df[targets]
        # print(X_df)
        # print(Y_df)
        df = X_df.join(Y_df, how="inner")
        df.sort("bayley_8_t2", inplace=True, ascending=True, kind="stable")

        if demo:
            take = min(df.shape[0] // 2, 30)
            df = pd.concat([df.iloc[:take], df.iloc[-take:]], axis="rows")
        if targetvar.endswith("t2"):
            agevar = "idade_crianca_dias_t2"
        elif targetvar.endswith("t3"):
            agevar = "idade_crianca_dias_t3"
        elif targetvar.endswith("t4"):
            agevar = "idade_crianca_dias_t4"
        else:
            raise Exception(f"Unexpected timepoint suffix for target '{targetvar}'.")
        # age = df[agevar]
        if noage:
            del df[agevar]  #####################################
        # df = df[["idade_crianca_dias_t2", "bayley_8_t2"]]
        # print(df.shape, "<<<<<<<<<<<<<<<<<")
        ret = loo(df, permutation=0, pairwise=pairwise, threshold=delta,
                  alg=alg, n_estimators=trees,
                  n_estimators_imp=trees_imp,
                  n_estimators_fsel=tfsel, forward_fsel=forward, k_features_fsel=feats, k_folds_fsel=4,
                  db=db, storages=storages, sched=sched,
                  seed=seed, jobs=jobs)
        if ret:
            d, bacc_c, hits_c, tot, tot_c, shap_c = ret
            print(f"\r{sp=} {delta=} {trees=} {bacc_c=:4.3f} | {hits_c=} {tot=} {tot_c=} \t{d.hosh.ansi} | {shap_c=} ", flush=True)

        # permutation test
        scores_dct = dict(bacc_c=[], bacc_r=[], r2_c=[], r2_r=[])
        for permutation in ap[1, 2, ..., perms]:
            df_shuffled = df.copy()
            df_shuffled[targetvar] = rnd.permutation(df[targetvar].values)
            ret = loo(df_shuffled, permutation, pairwise=pairwise, threshold=delta,
                      alg=alg, n_estimators=trees,
                      n_estimators_imp=trees_imp,
                      n_estimators_fsel=tfsel, forward_fsel=forward, k_features_fsel=feats, k_folds_fsel=4,
                      db=db, storages=storages, sched=sched,
                      seed=seed, jobs=jobs)
            if ret:
                d, bacc_cp, hits_cp, totp, tot_cp, shap_c = ret
                scores_dct["bacc_c"].append(bacc_cp - bacc_c)
                scores_dct["bacc_r"].append(bacc_rp - bacc_r)
                scores_dct["r2_c"].append(r2_cp - r2_c)
                scores_dct["r2_r"].append(r2_rp - r2_r)

        continue
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
