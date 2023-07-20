from germina.runner import run, run_t1_t2
from hdict import hdict

from sys import argv

loc = bool(int(argv[1]))
rem = bool(int(argv[2]))
nperm = int(argv[3])
nest = int(argv[4])
print("local cache:", loc)
print("remote cache:", rem)
print("permutations for p-value", nperm)
print("trees", nest)
print()
"""
        "elegib14_t0",  # sexo
"""
# "idade_crianca_meses_t1", "idade_crianca_meses_t2", "bisq_sleep_prob_t2"] # bisq_sleep_prob_t2/idade_crianca_meses_t2 mata 62 rows
d = hdict(n_permutations=nperm, n_splits=5, n_estimators=nest, random_state=0, index="id_estudo")
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
kwargs = dict(metavars=matts, loc=loc, rem=rem)

# run_t1_t2(d, microbiome=True, **kwargs)
# run_t1_t2(d, microbiome=True, mspecies=True, mpathways=True, msuper=True, **kwargs)
# run_t1_t2(d, microbiome=True, eeg=True, **kwargs)
# run_t1_t2(d, microbiome=True, eeg=True, eegpow=True, **kwargs)
# run_t1_t2(d, microbiome=True, mspecies=True, mpathways=True, msuper=True, eeg=True, **kwargs)
run_t1_t2(d, microbiome=True, mspecies=True, mpathways=True, msuper=True, eeg=True, eegpow=True, **kwargs)
