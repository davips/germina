from germina.runner import run
from hdict import hdict

"""
        "elegib14_t0",  # sexo
"""
matts = ["b13_t1", "maternal_ethinicity", "renda_familiar_total_t0", "a08_t1", "elegib2_t0", "c12f_t1", "idade_crianca_meses_t1", "infant_ethinicity", "EBF_3m", "delivery_mode", "chaos_tot_t1", "epds_2c_t1", "epds_tot_t1", "bisq_3_mins_t1", "bisq_4_mins_t1", "bisq_9_mins_t1", "bisq_sleep_prob_t1", "ebia_tot_t1", "ahmed_c14_2c_t1", "educationLevelAhmedNum_t1", "b04_t1", "a10_t1", "bmi_pregest_t1"]
# "idade_crianca_meses_t2", "bisq_sleep_prob_t2"] # mata 62 rows
mtgts = ["bayley_3_t1", "ibq_reg_t1", "ibq_soot_t2", "ibq_dura_t1"]
mtgts = ["ibq_reg_t1", "ibq_soot_t2", "ibq_dura_t1"]

d = hdict(n_permutations=10000, n_splits=5, n_estimators=500, random_state=0, index="id_estudo")

run(d, t1=True, t2=True, microbiome=True, eeg=True, metavars=matts, targets_meta=mtgts)
run(d, t1=True, t2=True, microbiome=True, eeg=True, metavars=matts,
    targets_eeg1=["Beta_t1", "r_20hz_post_pre_waveleting_t1", "Number_Segs_Post_Seg_Rej_t1"],
    targets_eeg2=["Beta_t2", "r_20hz_post_pre_waveleting_t2", "Number_Segs_Post_Seg_Rej_t2"])
