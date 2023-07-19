import sys
from pprint import pprint

import numpy as np
from lightgbm import LGBMClassifier as LGBMc
from numpy import array, quantile
from numpy import mean, std
from pandas import DataFrame
from sklearn import clone
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.inspection import permutation_importance
from sklearn.linear_model import SGDClassifier as SGDc
from sklearn.metrics import confusion_matrix, balanced_accuracy_score
from sklearn.metrics._scorer import balanced_accuracy_scorer
from sklearn.model_selection import KFold, cross_val_predict, StratifiedKFold, permutation_test_score
from statsmodels.stats.outliers_influence import variance_inflation_factor
from xgboost import XGBClassifier as XGBc

from germina.config import remote_cache_uri, local_cache_uri
from germina.dataset import join, ensemble_predict
from germina.nan import remove_cols, bina, loga, remove_nan_rows_cols, only_abundant
from hdict import _, apply, cache
from hdict import field
from hdict import hdict
from hdict.dataset.pandas_handling import file2df
from hosh import Hosh
from shelchemy import sopen

dropped = list({
                   '12DICHLORETHDEG-PWY': 0,
                   '1CMET2-PWY': 0,
                   'AEROBACTINSYN-PWY': 0,
                   'ALLANTOINDEG-PWY': 0,
                   'ANAEROFRUCAT-PWY': 0,
                   'ANAGLYCOLYSIS-PWY': 0,
                   'ARG+POLYAMINE-SYN': 0,
                   'ARGDEG-PWY': 0,
                   'ARGININE-SYN4-PWY': 0,
                   'ARGSYN-PWY': 0,
                   'ARGSYNBSUB-PWY': 0,
                   'ARO-PWY': 0,
                   'ASPASN-PWY': 0,
                   'AST-PWY': 0,
                   'BIOTIN-BIOSYNTHESIS-PWY': 0,
                   'BRANCHED-CHAIN-AA-SYN-PWY': 0,
                   'CALVIN-PWY': 0,
                   'CARNMET-PWY': 0,
                   'CATECHOL-ORTHO-CLEAVAGE-PWY': 0,
                   'CENTFERM-PWY': 0,
                   'CITRULBIO-PWY': 0,
                   'COA-PWY-1': 0,
                   'COA-PWY': 0,
                   'COBALSYN-PWY': 0,
                   'COLANSYN-PWY': 0,
                   'COMPLETE-ARO-PWY': 0,
                   'CRNFORCAT-PWY': 0,
                   'DAPLYSINESYN-PWY': 0,
                   'DARABCATK12-PWY': 0,
                   'DENITRIFICATION-PWY': 0,
                   'DENOVOPURINE2-PWY': 0,
                   'DHGLUCONATE-PYR-CAT-PWY': 0,
                   'DTDPRHAMSYN-PWY': 0,
                   'ECASYN-PWY': 0,
                   'FAO-PWY': 0,
                   'FASYN-ELONG-PWY': 0,
                   'FASYN-INITIAL-PWY': 0,
                   'FERMENTATION-PWY': 0,
                   'FOLSYN-PWY': 0,
                   'FUC-RHAMCAT-PWY': 0,
                   'FUCCAT-PWY': 0,
                   'GALACT-GLUCUROCAT-PWY': 0,
                   'GALACTARDEG-PWY': 0,
                   'GALACTITOLCAT-PWY': 0,
                   'GALACTUROCAT-PWY': 0,
                   'GLCMANNANAUT-PWY': 0,
                   'GLUCARGALACTSUPER-PWY': 0,
                   'GLUCONEO-PWY': 0,
                   'GLUCOSE1PMETAB-PWY': 0,
                   'GLUCUROCAT-PWY': 0,
                   'GLUDEG-I-PWY': 0,
                   'GLUTORN-PWY': 0,
                   'GLYCOCAT-PWY': 0,
                   'GLYCOGENSYNTH-PWY': 0,
                   'GLYCOL-GLYOXDEG-PWY': 0,
                   'GLYCOLYSIS-E-D': 0,
                   'GLYCOLYSIS-TCA-GLYOX-BYPASS': 0,
                   'GLYCOLYSIS': 0,
                   'GLYOXYLATE-BYPASS': 0,
                   'GOLPDLCAT-PWY': 0,
                   'HCAMHPDEG-PWY': 0,
                   'HEME-BIOSYNTHESIS-II-1': 0,
                   'HEME-BIOSYNTHESIS-II': 0,
                   'HEMESYN2-PWY': 0,
                   'HEXITOLDEGSUPER-PWY': 0,
                   'HISDEG-PWY': 0,
                   'HISTSYN-PWY': 0,
                   'HOMOSER-METSYN-PWY': 0,
                   'HSERMETANA-PWY': 0,
                   'ILEUSYN-PWY': 0,
                   'KDO-NAGLIPASYN-PWY': 0,
                   'KETOGLUCONMET-PWY': 0,
                   'LACTOSECAT-PWY': 0,
                   'LIPA-CORESYN-PWY': 0,
                   'LIPASYN-PWY': 0,
                   'LPSSYN-PWY': 0,
                   'MET-SAM-PWY': 0,
                   'METH-ACETATE-PWY': 0,
                   'METHGLYUT-PWY': 0,
                   'METSYN-PWY': 0,
                   'NAD-BIOSYNTHESIS-II': 0,
                   'NAGLIPASYN-PWY': 0,
                   'NONMEVIPP-PWY': 0,
                   'NONOXIPENT-PWY': 0,
                   'OANTIGEN-PWY': 0,
                   'ORNARGDEG-PWY': 0,
                   'ORNDEG-PWY': 0,
                   'P105-PWY': 0,
                   'P108-PWY': 0,
                   'P122-PWY': 0,
                   'P124-PWY': 0,
                   'P125-PWY': 0,
                   'P161-PWY': 0,
                   'P164-PWY': 0,
                   'P185-PWY': 0,
                   'P221-PWY': 0,
                   'P23-PWY': 0,
                   'P4-PWY': 0,
                   'P41-PWY': 0,
                   'P42-PWY': 0,
                   'P441-PWY': 0,
                   'P461-PWY': 0,
                   'P562-PWY': 0,
                   'P621-PWY': 0,
                   'PANTO-PWY': 0,
                   'PANTOSYN-PWY': 0,
                   'PENTOSE-P-PWY': 0,
                   'PEPTIDOGLYCANSYN-PWY': 0,
                   'PHOSLIPSYN-PWY': 0,
                   'POLYAMINSYN3-PWY': 0,
                   'POLYAMSYN-PWY': 0,
                   'POLYISOPRENSYN-PWY': 0,
                   'PPGPPMET-PWY': 0,
                   'PRPP-PWY': 0,
                   'PWY-1042': 0,
                   'PWY-1269': 0,
                   'PWY-1501': 0,
                   'PWY-1861': 0,
                   'PWY-2221': 0,
                   'PWY-241': 0,
                   'PWY-2941': 0,
                   'PWY-2942': 0,
                   'PWY-3001': 0,
                   'PWY-3801': 0,
                   'PWY-3841': 0,
                   'PWY-4041': 0,
                   'PWY-4361': 0,
                   'PWY-4984': 0,
                   'PWY-5004': 0,
                   'PWY-5005': 0,
                   'PWY-5022': 0,
                   'PWY-5028': 0,
                   'PWY-5030': 0,
                   'PWY-5088': 0,
                   'PWY-5097': 0,
                   'PWY-5100': 0,
                   'PWY-5103': 0,
                   'PWY-5104': 0,
                   'PWY-5121': 0,
                   'PWY-5130': 0,
                   'PWY-5136': 0,
                   'PWY-5138': 0,
                   'PWY-5154': 0,
                   'PWY-5156': 0,
                   'PWY-5180': 0,
                   'PWY-5188': 0,
                   'PWY-5189': 0,
                   'PWY-5265': 0,
                   'PWY-5345': 0,
                   'PWY-5347': 0,
                   'PWY-5367': 0,
                   'PWY-5384': 0,
                   'PWY-5392': 0,
                   'PWY-5415': 0,
                   'PWY-5417': 0,
                   'PWY-5431': 0,
                   'PWY-5464': 0,
                   'PWY-5484': 0,
                   'PWY-5497': 0,
                   'PWY-5505': 0,
                   'PWY-5531': 0,
                   'PWY-561': 0,
                   'PWY-5654': 0,
                   'PWY-5656': 0,
                   'PWY-5659': 0,
                   'PWY-5667': 0,
                   'PWY-5675': 0,
                   'PWY-5676': 0,
                   'PWY-5677': 0,
                   'PWY-5686': 0,
                   'PWY-5690': 0,
                   'PWY-5692': 0,
                   'PWY-5695': 0,
                   'PWY-5705': 0,
                   'PWY-5723': 0,
                   'PWY-5747': 0,
                   'PWY-5837': 0,
                   'PWY-5838': 0,
                   'PWY-5840': 0,
                   'PWY-5845': 0,
                   'PWY-5850': 0,
                   'PWY-5855': 0,
                   'PWY-5860': 0,
                   'PWY-5861': 0,
                   'PWY-5862': 0,
                   'PWY-5896': 0,
                   'PWY-5897': 0,
                   'PWY-5898': 0,
                   'PWY-5899': 0,
                   'PWY-5910': 0,
                   'PWY-5913': 0,
                   'PWY-5918': 0,
                   'PWY-5920': 0,
                   'PWY-5941': 0,
                   'PWY-5971': 0,
                   'PWY-5972': 0,
                   'PWY-5973': 0,
                   'PWY-5981': 0,
                   'PWY-5989': 0,
                   'PWY-6122': 4,
                   'PWY-7220': 105,
                   'PWY-8173': 173,
                   'PWY-8174': 173,
                   'PWY4FS-8': 211,
                   'PWY-7664': 149,
                   'PWY0-321': 200,
                   'UBISYN-PWY': 235,
                   'PWY-6126': 7,
                   'PWY0-1337': 189,
                   'PWY-6125': 6,
                   'PWY-6282': 19,
                   'PWY-7807': 153,
                   'PWY-6285': 20,
                   'PWY-7992': 161,
                   'PWY-6387': 29,
                   'PWY-7791': 149,
                   'TCA-GLYOX-BYPASS': 220,
                   'PWY0-1277': 177,
                   'PWY-6386': 28,
                   'PWY-7228': 101,
                   'PWY-6353': 26,
                   'PWY0-1415': 179,
                   'PWY-6182': 11,
                   'PWY-7229': 99,
                   'PWY-6519': 34,
                   'SALVADEHYPOX-PWY': 208,
                   'PWY0-845': 187,
                   'PWY0-1586': 179,
                   'VALSYN-PWY': 217,
                   'PWY0-166': 180,
                   'PWY0-301': 180,
                   'PWY-7560': 135,
                   'PWY-6277': 17,
                   'TRNA-CHARGING-PWY': 208,
                   'SO4ASSIM-PWY': 203,
                   'PWY-6385': 24,
                   'UDPNAGSYN-PWY': 207,
                   'PWY-7858': 143,
                   'THRESYN-PWY': 204,
                   'PWY-6901': 62,
                   'PWY0-1061': 162,
                   'PWY66-391': 186,
                   'PWY-6163': 9,
                   'PWY-6123': 4,
                   'PYRIDNUCSAL-PWY': 188,
                   'PWY-724': 98,
                   'PWY-7118': 77,
                   'PWY-7187': 79,
                   'PWY4FS-7': 176,
                   'PWY0-1338': 164,
                   'PWY-7357': 109,
                   'PWY1ZNC-1': 172,
                   'PWY-7204': 84,
                   'PWY-7388': 112,
                   'PWY-7117': 76,
                   'PWY-6969': 67,
                   'PWY-7663': 123,
                   'PWY0-1298': 156,
                   'PWY-841': 148,
                   'PWY-6531': 31,
                   'PWY-7238': 89,
                   'SULFATE-CYS-PWY': 179,
                   'PWY-7790': 125,
                   'PWY0-1477': 153,
                   'PWY-7111': 72,
                   'PWY-7208': 80,
                   'PWY-7391': 107,
                   'PWY-6121': 3,
                   'PWY0-1296': 146,
                   'PWY0-42': 152,
                   'PWY0-1319': 147,
                   'PWY-8178': 136,
                   'PWY-7282': 91,
                   'PWY-6803': 51,
                   'PWY-7222': 82,
                   'PWY-7268': 87,
                   'PWY-7953': 124,
                   'PWY-702': 66,
                   'PWY-6859': 53,
                   'PWY-7345': 93,
                   'PWY0-862': 144,
                   'PWY-6961': 62,
                   'PWY0-781': 142,
                   'PWY-6595': 36,
                   'PWY-6703': 45,
                   'PWY66-409': 146,
                   'PWY-8073': 121,
                   'PWY-6608': 38,
                   'PWY-6700': 43,
                   'PWY-7942': 115,
                   'PWY-7409': 98,
                   'PWY-6628': 40,
                   'PWY-7242': 76,
                   'PWY-801': 114,
                   'PWY-6284': 14,
                   'PWY-6305': 16,
                   'PWY-7269': 76,
                   'PWY-6527': 27,
                   'PWY-7723': 99,
                   'PWY0-1261': 120,
                   'PWY-7221': 70,
                   'PWY66-429': 131,
                   'PWY0-1479': 120,
                   'PWY-7874': 104,
                   'PWY-7013': 55,
                   'PWY-7234': 69,
                   'PWY-7199': 64,
                   'PWY-7400': 87,
                   'PYRIDOXSYN-PWY': 127,
                   'PWY-7197': 62,
                   'PWY0-1297': 113,
                   'PWY-6185': 8,
                   'PWY-7977': 99,
                   'PYRIDNUCSYN-PWY': 122,
                   'PWY-621': 10,
                   'PWY3O-4107': 114,
                   'PWY4LZ-257': 114,
                   'PWY-6895': 44,
                   'PWY-6147': 5,
                   'PWY-7115': 54,
                   'PWY-6606': 30,
                   'PWY-7211': 60,
                   'PWY-6936': 47,
                   'PWY-8187': 96,
                   'PWY-6435': 17,
                   'PWY-6609': 30,
                   'PWY-8004': 90,
                   'PWY-6151': 5,
                   'PWY-7094': 48,
                   'PWY-7184': 49,
                   'PWY0-41': 99,
                   'PWY-6317': 12,
                   'PWY0-1241': 95,
                   'PWY-6897': 38,
                   'PWY0-162': 95,
                   'RHAMCAT-PWY': 102,
                   'PWY-6612': 28,
                   'PWY-7851': 81,
                   'SER-GLYSYN-PWY': 102,
                   'PWY-6168': 5,
                   'PWY-6630': 27,
                   'PWY-6071': 1,
                   'PWY-7031': 40,
                   'PWY66-399': 93,
                   'PWY-7883': 78,
                   'PWY-6270': 7,
                   'THREOCAT-PWY': 96,
                   'PWY-7761': 72,
                   'PWY-7356': 56,
                   'PWY-6708': 26,
                   'PWY-6823': 30,
                   'PWY-7340': 53,
                   'PWY0-1221': 80,
                   'PWY-6545': 18,
                   'PWY-6293': 8,
                   'PWY-622': 6,
                   'PWY-6690': 22,
                   'PWY-7805': 65,
                   'PWY-6292': 6,
                   'PWY-6124': 2,
                   'PWY66-389': 77,
                   'PWY-7165': 32,
                   'PWY-7384': 48,
                   'PWY-6507': 12,
                   'PWY-6731': 19,
                   'PWY-7209': 33,
                   'REDCITCYC': 73,
                   'PWY-6590': 17,
               }.keys())


def calculate_vif(df: DataFrame, thresh=5.0):
    """https://stats.stackexchange.com/a/253620/36979"""
    X = df.assign(const=1)  # faster than add_constant from statsmodels
    # X = np.array(X, dtype=float)
    variables = list(range(X.shape[1]))
    dropped = True
    while dropped:
        dropped = False
        vif = [variance_inflation_factor(X.iloc[:, variables].values, ix)
               for ix in range(X.iloc[:, variables].shape[1])]
        vif = vif[:-1]  # don't let the constant be removed in the loop.
        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X.iloc[:, variables].columns[maxloc] +
                  '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped = True

    print('Remaining variables:')
    print(X.columns[variables[:-1]].to_list())
    return X.iloc[:, variables[:-1]]


def run(d: hdict, t1=False, t2=False, microbiome=False, microbiome_extra=False, eeg=False, metavars=None, targets_meta=None, targets_eeg1=None, targets_eeg2=None, stratifiedcv=True, path="data/", loc=True, rem=True):
    lst = []
    if t1:
        lst.append("t1")
    if t2:
        lst.append("t2")
    if microbiome:
        lst.append("bio")
    if microbiome_extra:
        lst.append("bio+")
    if eeg:
        lst.append("eeg")
    if metavars:
        lst.append(f"{metavars=}")
    if targets_meta:
        lst.append(f"{targets_meta=}")
    if targets_eeg1:
        lst.append(f"{targets_eeg1=}")
    if targets_eeg2:
        lst.append(f"{targets_eeg2=}")
    if stratifiedcv:
        lst.append("stratcv")
    name = "out/" + "§".join(lst).replace("_", "_").replace("§", "-") + ".txt"
    name = name.replace("=", "").replace("[", "«").replace("]", "»").replace(", ", ",").replace("'", "").replace("waveleting", "wv")
    name = name[:50] + Hosh(name.encode()).id
    print(name)
    oldout = sys.stdout
    with open(name, 'w') as sys.stdout:
        newout = sys.stdout
        sys.stdout = oldout

        print(f"Scenario: {t1=}, {t2=}, {microbiome=}, {microbiome_extra=}, {eeg=},\n"
              f"{metavars=},\n"
              f"{targets_meta=},\n"
              f"{targets_eeg1=},\n"
              f"{targets_eeg2=}")
        pprint(dict(d))
        print()
        d = d >> dict(join="inner", shuffle=True, n_jobs=-1, return_name=False)

        if metavars is None:
            metavars = []
        if targets_meta is None:
            targets_meta = []
        if targets_eeg2 is None:
            targets_eeg2 = []
        if targets_eeg1 is None:
            targets_eeg1 = []
        targets = targets_meta + targets_eeg1 + targets_eeg2
        with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
            if microbiome:  #################################################################################################################
                if t1:
                    d = d >> apply(file2df, path + "data_microbiome___2023-06-18___alpha_diversity_n525.csv").microbiome_alpha1
                    if microbiome_extra:
                        d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").microbiome_pathways1
                        d = d >> apply(only_abundant, _.microbiome_pathways1).microbiome_pathways1
                        d = d >> apply(file2df, path + "data_microbiome___2023-06-18___especies_3_meses_n525.csv").microbiome_species1
                        d = d >> apply(only_abundant, _.microbiome_species1).microbiome_species1
                        d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T1_vias_relab_superpathways.csv").microbiome_super1
                if t2:
                    d = d >> apply(file2df, path + "data_microbiome___2023-07-03___alpha_diversity_T2_n441.csv").microbiome_alpha2
                    if microbiome_extra:
                        d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").microbiome_pathways2
                        d = d >> apply(only_abundant, _.microbiome_pathways2).microbiome_pathways2
                        d = d >> apply(file2df, path + "data_microbiome___2023-06-29___especies_6_meses_n441.csv").microbiome_species2
                        d = d >> apply(only_abundant, _.microbiome_species2).microbiome_species2
                        d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T2_vias_relab_superpathways.csv").microbiome_super2

            if eeg:  ########################################################################################################################
                if (t1 and not targets_eeg2) or targets_eeg1:
                    d = d >> apply(file2df, path + "data_eeg___2023-06-20___T1_RS_average_dwPLI_withEEGCovariates.csv").eeg1
                    d = d >> apply(file2df, path + "data_eeg___2023-07-19___BRAINRISE_RS_3m_power.csv").eegpow1
                if t2 or targets_eeg2:
                    d = d >> apply(file2df, path + "data_eeg___2023-06-20___T2_RS_average_dwPLI_withEEGCovariates.csv").eeg2
                    d = d >> apply(file2df, path + "data_eeg___2023-07-19___BRAINRISE_RS_T2_Power.csv").eegpow2
                if targets_eeg1:
                    d = d >> apply(DataFrame.__getitem__, _.eeg1, ["id_estudo"] + targets_eeg1).eeg1
                if targets_eeg2:
                    d = d >> apply(DataFrame.__getitem__, _.eeg2, ["id_estudo"] + targets_eeg2).eeg2

            # join #######################################################################################################################
            if microbiome:
                if t1:
                    d["df"] = _.microbiome_alpha1
                    if microbiome_extra:
                        d = d >> apply(join, other=_.microbiome_pathways1).df
                        d = d >> apply(join, other=_.microbiome_species1).df
                        d = d >> apply(join, other=_.microbiome_super1).df
                if t2:
                    if "df" not in d:
                        d["df"] = _.microbiome_alpha2
                    else:
                        d = d >> apply(join, other=_.microbiome_alpha2).df
                    if microbiome_extra:
                        d = d >> apply(join, other=_.microbiome_pathways2).df
                        d = d >> apply(join, other=_.microbiome_species2).df
                        d = d >> apply(join, other=_.microbiome_super2).df
            if eeg or targets_eeg1 or targets_eeg2:
                if (t1 and not targets_eeg2) or targets_eeg1:
                    if "df" not in d:
                        d["df"] = _.eeg1
                    else:
                        d = d >> apply(join, other=_.eeg1).df
                    if "eegpow1" in d and not (targets_eeg1 or targets_eeg2):
                        d = d >> apply(join, other=_.eegpow1).df
                if t2 or targets_eeg2:
                    if "df" not in d:
                        d["df"] = _.eeg2
                    else:
                        d = d >> apply(join, other=_.eeg2).df
                    if "eegpow2" in d and not (targets_eeg1 or targets_eeg2):
                        d = d >> apply(join, other=_.eegpow2).df
            # d = d >> apply(remove_nan_rows_cols, cols_at_a_time=0, keep=["id_estudo"] + targets).df
            if rem:
                d = d >> cache(remote)
            if loc:
                d = d >> cache(local)
            print("Joined------------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Join metadata #############################################################################################################
            if metavars:
                d = d >> apply(file2df, path + "metadata___2023-07-17.csv").metadata
                d = d >> apply(DataFrame.__getitem__, _.metadata, metavars + ["id_estudo"]).metadata
                print("Format problematic attributes.")
                d = d >> apply(bina, _.metadata, attribute="antibiotic", positive_category="yes").metadata
                d = d >> apply(bina, _.metadata, attribute="EBF_3m", positive_category="EBF").metadata
                d = d >> apply(loga, _.metadata, attribute="renda_familiar_total_t0").metadata
                d = d >> apply(join, other=_.metadata).df
                d = d >> apply(remove_nan_rows_cols, keep=["id_estudo"] + targets).df
                if rem:
                    d = d >> cache(remote)
                if loc:
                    d = d >> cache(local)
                print("Metadata----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")
            # d.df.to_csv(f"/tmp/all.csv")
            # exit()

            d = d >> apply(remove_cols, cols=dropped, keep=[]).df
            d = d >> apply(calculate_vif).df
            if rem:
                d = d >> cache(remote)
            if loc:
                d = d >> cache(local)

            # Join targets ##############################################################################################################
            if targets_meta:
                d = d >> apply(file2df, path + "metadata___2023-06-18.csv").targets
                d = d >> apply(DataFrame.__getitem__, _.targets, targets + ["id_estudo"]).targets
                d = d >> apply(join, other=_.targets).df
                if rem:
                    d = d >> cache(remote)
                if loc:
                    d = d >> cache(local)
            print("Dataset-----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Remove NaNs ##################################################################################################################
            d = d >> apply(remove_nan_rows_cols, keep=["id_estudo"] + targets).df
            print("Dataset without NaNs ------------------------------------------------------------\n", d.df, "______________________________________________________\n")

            # Visualize ####################################################################################################################
            print("Vars:", d.df.columns.to_list())
            # d.df.to_csv(f"/tmp/all.csv")
            # d.df: DataFrame
            # for target in targets:
            #     d.df[target].hist(bins=3)
            # plt.show()
            #

            # Train #######################################################################################################################
            if stratifiedcv:
                d = d >> apply(StratifiedKFold).cv
            else:
                d = d >> apply(KFold).cv
            for target in targets:
                print("=======================================================")
                print(target)
                print("=======================================================")

                # Prepare dataset.
                d = d >> apply(getattr, _.df, target).t
                d = d >> apply(lambda x: np.digitize(x, quantile(x, [1 / 5, 4 / 5])), _.t).t
                d = d >> apply(lambda df, t: df[t != 1], _.df, _.t).dfcut
                d = d >> apply(remove_cols, _.dfcut, targets, keep=[]).X
                d = d >> apply(lambda t: t[t != 1]).t
                d = d >> apply(lambda t: t // 2).y

                if rem:
                    d = d >> cache(remote)
                if loc:
                    d = d >> cache(local)
                print("X:", d.X.shape)
                print("y:", d.y.shape)

                clas_names = []
                clas = {
                    # DummyClassifier: {},
                    RFc: {},
                    XGBc: {},
                    # CATc: {"subsample": 0.1},
                    LGBMc: {},
                    ETc: {},
                    SGDc: {},
                }
                for cla, kwargs in clas.items():
                    clas_names.append(cla.__name__)
                    # print(clas_names[-1])
                    d = d >> apply(cla, **kwargs)(clas_names[-1])

                # Prediction power.
                ['accuracy', 'adjusted_mutual_info_score', 'adjusted_rand_score', 'average_precision',
                 'balanced_accuracy', 'completeness_score', 'explained_variance',
                 'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted', 'fowlkes_mallows_score',
                 'homogeneity_score', 'jaccard', 'jaccard_macro', 'jaccard_micro', 'jaccard_samples', 'jaccard_weighted',
                 'matthews_corrcoef', 'max_error', 'mutual_info_score',
                 'neg_brier_score', 'neg_log_loss', 'neg_mean_absolute_error', 'neg_mean_absolute_percentage_error', 'neg_mean_gamma_deviance', 'neg_mean_poisson_deviance', 'neg_mean_squared_error', 'neg_mean_squared_log_error', 'neg_median_absolute_error', 'neg_negative_likelihood_ratio', 'neg_root_mean_squared_error',
                 'normalized_mutual_info_score', 'positive_likelihood_ratio',
                 'precision', 'precision_macro', 'precision_micro', 'precision_samples', 'precision_weighted',
                 'r2', 'rand_score',
                 'recall', 'recall_macro', 'recall_micro', 'recall_samples', 'recall_weighted',
                 'roc_auc', 'roc_auc_ovo', 'roc_auc_ovo_weighted', 'roc_auc_ovr', 'roc_auc_ovr_weighted', 'top_k_accuracy', 'v_measure_score']
                scos = ["precision", "recall", "balanced_accuracy", "roc_auc"]
                scos = ["roc_auc", "balanced_accuracy"]
                for m in scos:
                    print(m)
                    print("vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv")
                    for classifier_field in clas_names:
                        scores_fi = f"{m}_{classifier_field}"
                        permscores_fi = f"perm_{scores_fi}"
                        pval_fi = f"pval_{scores_fi}"
                        # d = d >> apply(cross_val_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(scores_fi)
                        d = d >> apply(permutation_test_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(scores_fi, permscores_fi, pval_fi)
                        if rem:
                            d = d >> cache(remote)
                        if loc:
                            d = d >> cache(local)
                        me = mean(d[scores_fi])
                        if classifier_field == "DummyClassifier":
                            ref = me
                        print(f"{classifier_field:24} {me:.6f} {std(d[scores_fi]):.6f}   p-value={d[pval_fi]}")
                    print("^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^")

                # ConfusionMatrix; prediction and hit agreement.
                zs, hs = {}, {}
                members_z = []
                for classifier_field in clas_names:
                    print(classifier_field)
                    field_name_z = f"{classifier_field}_z"
                    if not classifier_field.startswith("Dummy"):
                        members_z.append(field(field_name_z))
                    d = d >> apply(cross_val_predict, field(classifier_field), _.X, _.y, cv=_.cv)(field_name_z)
                    if rem:
                        d = d >> cache(remote)
                    if loc:
                        d = d >> cache(local)
                    z = d[field_name_z]
                    zs[classifier_field[:10]] = z
                    hs[classifier_field[:10]] = (z == d.y).astype(int)
                    print(f"{confusion_matrix(d.y, z)}")
                d = d >> apply(ensemble_predict, *members_z).ensemble_z
                if rem:
                    d = d >> cache(remote)
                if loc:
                    d = d >> cache(local)

                # Accuracy
                for classifier_field in clas_names:
                    field_name_z = f"{classifier_field}_z"
                    fieldbalacc = f"{classifier_field}_balacc"
                    d = d >> apply(balanced_accuracy_score, _.y, field(field_name_z), adjusted=True)(fieldbalacc)
                    if rem:
                        d = d >> cache(remote)
                    if loc:
                        d = d >> cache(local)
                    print(f"{classifier_field:24} {d[fieldbalacc]:.6f} ")
                d = d >> apply(balanced_accuracy_score, _.y, _.ensemble_z, adjusted=True).ensemble_balacc
                if rem:
                    d = d >> cache(remote)
                if loc:
                    d = d >> cache(local)
                print(f"ensemble5 {d.ensemble_balacc:.6f} ")

                print("Prediction:")
                Z = array(list(zs.values()))
                zs["   AND    "] = np.logical_and.reduce(Z, axis=0).astype(int)
                zs["   OR     "] = np.logical_or.reduce(Z, axis=0).astype(int)
                zs["   SUM    "] = np.sum(Z, axis=0).astype(int)
                zs["   NOR    "] = np.logical_not(np.logical_or.reduce(Z, axis=0)).astype(int)
                zs["   ==     "] = (np.logical_and.reduce(Z, axis=0) | np.logical_not(np.logical_or.reduce(Z, axis=0))).astype(int)
                for k, z in zs.items():
                    if "AND" in k:
                        print()
                    # print(k, sum(z), ",".join(map(str, z)))
                print()
                print("Hit:")
                H = array(list(hs.values()))
                hs["   AND    "] = np.logical_and.reduce(H, axis=0).astype(int)
                hs["   OR     "] = np.logical_or.reduce(H, axis=0).astype(int)
                hs["   SUM    "] = np.sum(H, axis=0).astype(int)
                hs["   NOR    "] = np.logical_not(np.logical_or.reduce(H, axis=0)).astype(int)
                hs["   ==     "] = (np.logical_and.reduce(H, axis=0) | np.logical_not(np.logical_or.reduce(H, axis=0))).astype(int)
                for k, h in hs.items():
                    if "AND" in k:
                        print()
                    # print(k, sum(h), "\t", ",".join(map(str, h)))
                print()

                # Importances
                for classifier_field in clas_names:
                    model = f"{target}_{classifier_field}_model"
                    d = d >> apply(lambda c, *args, **kwargs: clone(c).fit(*args, **kwargs), field(classifier_field), _.X, _.y)(model)
                    importances_field_name = f"{target}_{classifier_field}_importances"
                    d = d >> apply(permutation_importance, field(model), _.X, _.y, n_repeats=20, scoring=scos, n_jobs=-1)(importances_field_name)
                    if rem:
                        d = d >> cache(remote)
                    if loc:
                        d = d >> cache(local)
                    fst = True
                    for metric in d[importances_field_name]:
                        r = d[importances_field_name][metric]
                        for i in r.importances_mean.argsort()[::-1]:
                            if r.importances_mean[i] - r.importances_std[i] > 0:
                                if fst:
                                    print(f"Importances {classifier_field:<20} ----------------------------")
                                    fst = False
                                print(f"  {metric:<17} {d.X.columns[i][-25:]:<17} {r.importances_mean[i]:.6f} +/- {r.importances_std[i]:.6f}")
                        # if not fst:
                        #     print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                        print()
                print()
        # sys.stdout = oldout
        # d.show()
        # sys.stdout = newout

    sys.stdout = oldout
