from pprint import pprint

from hdict import hdict
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sys import argv

from hdict import hdict
from shelchemy import sopen
from shelchemy.scheduler import Scheduler
from sklearn.model_selection import StratifiedKFold

from germina.config import schedule_uri
from germina.runner import run_t1_t2
from argvsucks import handle_command_line

dct = handle_command_line(argv, loc=False, rem=False, permutations=int, trees=int, stacking=False, strees=int, ssplits=int, sync=False, measures=list, schedule=False, print=False)
loc, rem, permutations, trees, stacking, strees, ssplits, sync, measures, schedule, printing = dct["loc"], dct["rem"], dct["permutations"], dct["trees"], dct["stacking"], dct["strees"], dct["ssplits"], dct["sync"], dct["measures"], dct["schedule"], dct["print"]
pprint(dct)
print()
"""
        "elegib14_t0",  # sexo
"""
# "idade_crianca_meses_t1", "idade_crianca_meses_t2", "bisq_sleep_prob_t2"] # bisq_sleep_prob_t2/idade_crianca_meses_t2 mata 62 rows
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
d = hdict(n_permutations=permutations, n_splits=5, n_estimators=trees,
          stacking=stacking, stacking_cv=StratifiedKFold(n_splits=ssplits), stacking_final_estimator=RandomForestClassifier(n_estimators=strees),
          measures=measures,
          random_state=0, index="id_estudo")

kwargs0 = dict(metavars=matts, loc=loc, rem=rem, sync=sync, scheduler=schedule, printing=printing)
mbioma = dict(malpha=True, mspecies=True, mpathways=True, msuper=True)
eeg = dict(eeg=True, eegpow=True)

kwargs = kwargs0 | mbioma | eeg
d["kwargs"] = kwargs
del dct["loc"]
del dct["rem"]
del dct["stacking"]
del dct["sync"]
del dct["schedule"]
del dct["print"]
del dct["measures"]
d["dct"] = dct
kwargs["did"] = d.id
if "empty_mbioma" in kwargs:
    del kwargs["empty_mbioma"]
if "empty_eeg" in kwargs:
    del kwargs["empty_eeg"]
run_t1_t2(d, **kwargs)
