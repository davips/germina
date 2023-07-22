import graphviz
import numpy as np
from hdict import hdict, apply
from numpy import mean
from pandas import DataFrame
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.tree import DecisionTreeClassifier, export_graphviz, DecisionTreeRegressor

from germina.config import schedule_uri
from germina.nan import remove_cols, select_cols
from germina.runner import run_t1_t2, run

matts = [
    "b13_t1",  # father ethnicity
    "maternal_ethinicity", "b04_t1",  # mother ethnicity
    "renda_familiar_total_t0",
    "infant_ethinicity", "a08_t1",  # infant ethnicity
    "elegib2_t0",  # mother age
    "c12f_t1",  # depression during or posparto
    "EBF_3m", "delivery_mode",
    "chaos_tot_t1",  # Confusion, hubbub, and order scale
    "epds_2c_t1",  # EPDS Classification according to Santos et al. 2007
    "epds_tot_t1",
    "bisq_3_mins_t1",  # nocturnal sleep
    "bisq_4_mins_t1",  # diurnal sleep
    "bisq_9_mins_t1",  # nocturnal sleep time
    "bisq_sleep_prob_t1",  # any sleep problem according to Sadeh 2004
    "ebia_tot_t1",  # Food insecurity
    "ahmed_c14_2c_t1", "educationLevelAhmedNum_t1",
    "a10_t1",  # ordem desta criança entre os irmãos
    "bmi_pregest_t1"
]
attributes1 = ["chaos_tot_t1", "Frontocentral_delta_t1",
               "k__Bacteria|p__Actinobacteria|c__Actinobacteria|o__Bifidobacteriales|f__Bifidobacteriaceae|g__Bifidobacterium|s__Bifidobacterium_dentium",
               "PWY-7456", "THISYNARA-PWY", "PWY-7328", "DTDPRHAMSYN-PWY", "PWY-7237", "Theta_t1", "PWY66-430", "PWY-6749", "H_lateral_frontal_beta_t1",
               "HighAlpha_t1", "PWY-6902", "P164-PWY", "bisq_4_mins_t1"]
attributes2 = ["CENTFERM-PWY_t2", "HighAlpha_t2", "AEROBACTINSYN-PWY_t2", "LH_lateral_frontal_beta_t1", "HighAlpha_t1", "bisq_4_mins_t1"]
targets_meta1 = ['ibq_reg_t1', 'ibq_soot_t1', 'ibq_dura_t1', 'bayley_3_t1']
targets_meta2 = ['ibq_reg_t2', 'ibq_soot_t2', 'ibq_dura_t2', 'bayley_3_t2']
targets_eeg1 = ["Beta_t1", "r_20hz_post_pre_waveleting_t1", "Number_Segs_Post_Seg_Rej_t1"]
targets_eeg2 = ["Beta_t2", "r_20hz_post_pre_waveleting_t2", "Number_Segs_Post_Seg_Rej_t2"]

d = hdict(random_state=0, index="id_estudo")
kwargs = dict(just_df=True, did=d.id, metavars=matts, malpha=True, mspecies=True, mpathways=True, msuper=True, eeg=True, eegpow=True)

datasetses = {
    "t1 → t1 meta": run(d, t1=True, targets_meta=targets_meta1, **kwargs),
    "t1 → t1 eeg": run(d, t1=True, targets_eeg1=targets_eeg1, **kwargs),

    "t1 → t2 meta": run(d, t1=True, targets_meta=targets_meta2, **kwargs),
    "t1 → t2 eeg": run(d, t1=True, targets_eeg2=targets_eeg2, **kwargs),

    "t1+t2 → t2 meta": run(d, t1=True, t2=True, targets_meta=targets_meta2, **kwargs),
    "t1+t2 → t2 eeg": run(d, t1=True, t2=True, targets_eeg2=targets_eeg2, **kwargs)
}

for t, datasets in datasetses.items():
    selected = attributes1
    if "+t2" in t:
        selected += attributes2
    for target, (X, y) in datasets.items():

        print(target, list(X.columns))
        X = select_cols(X, selected)
        print(target, list(X.columns))

        algs = [
            DecisionTreeClassifier(max_depth=4, min_samples_split=10, min_samples_leaf=16, min_impurity_decrease=0.01, random_state=0, max_leaf_nodes=8),
            # DecisionTreeRegressor(max_depth=3, min_samples_split=22, min_samples_leaf=10, random_state=0, max_leaf_nodes=8)
        ]
        ys = [y]
        print(f"{target + ',':16}", end="")
        for alg, y in zip(algs, ys):
            cla = alg.__class__.__name__
            model = alg.fit(X, y)
            dot_data = export_graphviz(model, out_file=None, feature_names=X.columns, class_names=sorted(str(x) for x in np.unique(y)), filled=True)
            graph = graphviz.Source(dot_data)
            arq = f"png/{cla}-{len(X.columns)}attrs-" + target
            graph.render(view=False, filename=arq, format="png", cleanup=True)
            print(f"{arq}", end="\t")
        print()
