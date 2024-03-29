from pprint import pprint

from lightgbm import LGBMClassifier as LGBMc, LGBMRegressor as LGBMr
import pandas as pd
from lange import ap, gp
from matplotlib import pyplot as plt
from pandas import read_csv, DataFrame
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
from sklearn.model_selection import permutation_test_score, LeaveOneOut, StratifiedKFold
from xgboost import XGBClassifier
import numpy as np

a = [x.split("\t")[0] for x in """Veillonella_sp_T11011_6	4.6%
Klebsiella_aerogenes	4.9%
Morganella_morganii	4.9%
Bilophila_wadsworthia	5.3%
Veillonella_rogosae	5.3%
Megasphaera_micronuciformis	5.6%
Raoultella_ornithinolytica	5.6%
Clostridium_clostridioforme	6.0%
Cutibacterium_avidum	6.0%
Enterococcus_casseliflavus	6.0%
Anaerostipes_caccae	6.3%
Streptococcus_vestibularis	6.3%
Streptococcus_lutetiensis	7.0%
Veillonella_seminalis	7.0%
Enterococcus_avium	7.4%
Ruthenibacterium_lactatiformans	7.7%
Bacteroides_caccae	8.1%
Bacteroides_stercoris	8.1%
Clostridium_bolteae	8.1%
Enterococcus_gallinarum	8.1%
Parabacteroides_merdae	8.1%
Bifidobacterium_scardovii	8.5%
Haemophilus_sp_HMSC71H05	8.5%
Rothia_mucilaginosa	8.5%
Staphylococcus_epidermidis	8.5%
Bifidobacterium_adolescentis	8.8%
Veillonella_infantium	9.2%
Enterococcus_faecium	9.9%
Bacteroides_dorei	10.2%
Clostridium_sp_7_2_43FAA	10.2%
Bacteroides_ovatus	10.6%
Intestinibacter_bartlettii	10.9%
Haemophilus_haemolyticus	11.3%
Bacteroides_thetaiotaomicron	11.6%
Lactobacillus_casei_group	11.6%
Hungatella_hathewayi	12.3%
Proteus_mirabilis	12.3%
Bacteroides_fragilis	12.7%
Klebsiella_oxytoca	12.7%
Campylobacter_concisus	13.0%
Bifidobacterium_pseudocatenulatum	13.4%
Citrobacter_freundii	13.4%
Clostridium_innocuum	14.1%
Veillonella_sp_DORA_A_3_16_22	14.1%
Bacteroides_uniformis	14.4%
Collinsella_aerofaciens	15.5%
Staphylococcus_aureus	15.8%
Lactobacillus_rhamnosus	16.2%
Bifidobacterium_breve	16.5%
Clostridium_butyricum	16.9%
Lactobacillus_gasseri	17.3%
Enterobacter_cloacae_complex	18.7%
Bifidobacterium_bifidum	19.0%
Bacteroides_vulgatus	19.7%
Clostridioides_difficile	19.7%
Parabacteroides_distasonis	19.7%""".split("\n")]

b = [x.split("\t")[0] for x in """Eggerthella_lenta	22.9%
Klebsiella_michiganensis	23.6%
Flavonifractor_plautii	27.5%
Clostridium_paraputrificum	28.9%
Bifidobacterium_dentium	30.6%
Streptococcus_mitis	30.6%
Erysipelatoclostridium_ramosum	32.7%
Streptococcus_parasanguinis	32.7%
Ruminococcus_gnavus	34.2%
Veillonella_atypica	37.3%
Haemophilus_parainfluenzae	38.4%
Clostridium_perfringens	39.8%""".split("\n")]

c = [x.split("\t")[0] for x in """Klebsiella_quasipneumoniae	43.3%
Bifidobacterium_longum	43.7%
Veillonella_dispar	44.4%
Clostridium_neonatale	50.0%
Klebsiella_variicola	52.1%
Enterococcus_faecalis	52.8%
Veillonella_parvula	54.9%
Streptococcus_salivarius	60.2%
Klebsiella_pneumoniae	63.4%
Escherichia_coli	78.9%""".split("\n")]
ax = axd = None
rejcolor = "gray"
for sp in [1, 2]:
    scolab = f"Balanced Accuracy T{sp}"
    df = read_csv(f"/home/davi/git/germina/results/datasetr_species{sp}_bayley_8_t2.csv", index_col="id_estudo")
    # df.drop(a, axis="columns", inplace=True)
    # df.drop(b, axis="columns", inplace=True)
    print(df.shape)
    # print(df.shape)
    # print("---------------")
    # df.sort_values("idade_crianca_dias_t2", inplace=True)  # age at bayley test
    age = df["idade_crianca_dias_t2"]
    yr = df["bayley_8_t2"]

    # hiidx = df.index[yr >= 100.0]  # non arbitrary scale-based
    # loidx = df.index[yr < 100.0]
    # hiidx = df.index[yr >= 107.5]  # scale-based
    # loidx = df.index[yr <= 92.5]
    hiidx = df.index[yr >= 107.86]  # sample-based
    loidx = df.index[yr <= 99.06]

    print("sp:", sp, "balance:", len(loidx), len(hiidx))
    X = pd.concat([df.loc[loidx], df.loc[hiidx]])
    del X["bayley_8_t2"]
    # del X["idade_crianca_dias_t2"]
    hiy = yr[hiidx].astype(int) * 0 + 1
    loy = yr[loidx].astype(int) * 0
    y0 = pd.concat([loy, hiy])
    X0 = X

    trees = 64
    r = {0: 0, 1: 0}
    t = r.copy()
    lst = []
    newindex = []
    for idx in X.index:
        X = X0.drop(idx, axis="rows")
        y = y0.drop(idx, axis="rows")
        alg = LGBMc(n_estimators=trees, n_jobs=1)
        # alg = LGBMr(n_estimators=trees, n_jobs=1)
        # score, permscores, pval = permutation_test_score(alg, X=X, y=y, cv=StratifiedKFold(n_splits=10), scoring="r2", n_permutations=1, n_jobs=1)
        alg.fit(X, y)
        x = X0.loc[[idx], :]
        label = y0.loc[idx]
        prediction = alg.predict(x)[0]
        prob = np.max(alg.predict_proba(x))
        newindex.append(idx)
        lst.append([prob, label, prediction, int(label == prediction)])
        if prediction == label:
            r[label] += 1
        t[label] += 1

    # sem "nao sei"
    acc0 = r[0] / t[0]
    acc1 = r[1] / t[1]
    score = (acc0 + acc1) / 2
    print(f"\t", trees, "score:", score, sep="\t")
    print("---------------------")

    # continue

    # com "nao sei"
    res = DataFrame(lst, columns=["prob", "label", "prediction", "hit"], index=newindex)
    # print(res)
    lst = []
    for rejection_width in ap[0.01, 0.02, ..., 1]:
        res2 = res[abs(res["prob"] - 0.5) > rejection_width / 2]
        n = len(res2)
        if res2.empty:
            continue

        r = {0: 0, 1: 0}
        t = r.copy()
        for idx in res2.index:
            re = res2.loc[idx]
            label = re["label"]
            if re["hit"]:
                r[label] += 1
            t[label] += 1
        if 0 in t.values():
            continue
        acc0 = r[0] / t[0]
        acc1 = r[1] / t[1]
        score = (acc0 + acc1) / 2
        # print(f"\tP>{cutoff}\t{res2.shape}\t{trees} trees score:", score, sep="\t")
        lst.append([rejection_width, 100 * score, 100 * (len(res) - n) / len(res)])
    nlab = f"Rejected Instances T{sp}"
    cusco = DataFrame(lst, columns=["Rejection Width", scolab, nlab])
    ax = cusco.plot(x="Rejection Width", y=scolab, kind="line", ax=ax, ylabel="Balanced Accuracy")
    ax.legend(bbox_to_anchor=(0.3, 1.15))
    if axd is None:
        axd = ax.twinx()
    else:
        rejcolor = "black"
    cusco.plot(x="Rejection Width", y=nlab, kind="line", ax=axd, c=rejcolor, ylabel="Rejected Instances (%)")
    axd.legend(bbox_to_anchor=(1.1, 1.15))
    print("------------------------------------------")
    print()

plt.show()
