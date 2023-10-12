from pprint import pprint

from germina.dataset import join, metavars_no_target, vif_dropped
from pandas import DataFrame

from germina.runner import drop_many_by_vif, ch

from germina.nan import only_abundant, remove_nan_rows_cols, bina, loga, remove_cols
from hdict import apply, hdict, _, field
from hdict.dataset.pandas_handling import file2df

from germina.config import local_cache_uri
from shelchemy import sopen

path = "data/"
target = "bayley_3_t4"
loc, rem, remote = True, False, None
sync = False
vif = True
# bayley_average_t4
d = hdict(target="ibq_reg_cat_t3", index="id_estudo", join="inner", shuffle=True, n_jobs=-1, return_name=False)

with sopen(local_cache_uri) as local:
    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T1_n525.csv").microbiome_pathways1
    d = d >> apply(only_abundant, _.microbiome_pathways1).microbiome_pathways1
    if vif:
        d = d >> apply(remove_cols, _.microbiome_pathways1, cols=vif_dropped, keep=[], debug=False).microbiome_pathways1
        d = drop_many_by_vif(d, "microbiome_pathways1", loc, rem, local, remote, sync)

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T1_vias_relab_superpathways.csv").microbiome_super1
    if vif:
        d = d >> apply(remove_cols, _.microbiome_super1, cols=vif_dropped, keep=[], debug=False).microbiome_super1
        d = drop_many_by_vif(d, "microbiome_super1", loc, rem, local, remote, sync)

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___vias_metabolicas_valor_absoluto_T2_n441.csv").microbiome_pathways2
    d = d >> apply(only_abundant, _.microbiome_pathways2).microbiome_pathways2
    if vif:
        d = d >> apply(remove_cols, _.microbiome_pathways2, cols=vif_dropped, keep=[], debug=False).microbiome_pathways2
        d = drop_many_by_vif(d, "microbiome_pathways2", loc, rem, local, remote, sync)

    d = d >> apply(file2df, path + "data_microbiome___2023-07-04___T2_vias_relab_superpathways.csv").microbiome_super2
    if vif:
        d = d >> apply(remove_cols, _.microbiome_super2, cols=vif_dropped, keep=[], debug=False).microbiome_super2
        d = drop_many_by_vif(d, "microbiome_super2", loc, rem, local, remote, sync)
    d = ch(d, loc, rem, local, remote, sync)
    d.show()

    # Join all non OSF data #####################################################################################################
    d["df"] = _.microbiome_pathways1
    d = d >> apply(join, other=_.microbiome_super1).df
    d = d >> apply(join, other=_.microbiome_pathways2).df
    d = d >> apply(join, other=_.microbiome_super2).df
    d = ch(d, loc, rem, local, remote, sync)
    print("Joined------------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

    # Join metadata #############################################################################################################
    d = d >> apply(file2df, path + "germina-osf-request---davi121023.csv").metadata_full
    metavars = ["id_estudo"]
    for v in sorted(set(metavars_no_target).difference(vif_dropped)):
        for i in range(7):
            sub = f"{v}_t{i}"
            if sub in d.metadata_full:
                metavars.append(sub)
    metavars.sort()
    d = d >> apply(lambda metadata_full, mtvs: metadata_full[mtvs], mtvs=metavars).metadata

    print("Format problematic attributes.")
    # d = d >> apply(bina, _.metadata, attribute="antibiotic", positive_category="yes").metadata
    # d = d >> apply(bina, _.metadata, attribute="EBF_3m", positive_category="EBF").metadata
    # for i in range(7):
    #     d = d >> apply(loga, _.metadata, attribute=f"renda_familiar_total_t{i}").metadata
    d = d >> apply(join, other=_.metadata).df
    # d = d >> apply(remove_nan_rows_cols, keep=["id_estudo", target]).df
    d = ch(d, loc, rem, local, remote, sync)
    print("With metadata----------------------------------------------------------\n", d.df, "______________________________________________________\n")

    ##############################   VIF    #####################################################################################
    if vif:
        pprint([d.hosh, d.hoshes])
        d = drop_many_by_vif(d, "df", loc, rem, local, remote, sync)

    # Join targets ##############################################################################################################
    d = d >> apply(lambda metadata_full, target: metadata_full[target, "id_estudo"].reindex(sorted([target, "id_estudo"]), axis=1)).target

    d = d >> apply(join, other=_.target).df
    d = ch(d, loc, rem, local, remote, sync)
    print("Dataset-----------------------------------------------------------------------\n", d.df, "______________________________________________________\n")

    print("Restart now by using only noncolinear columns ----------------------------------------------------")
    d = d >> apply(lambda df: df.columns.to_list()).columns
    d = ch(d, loc, rem, local, remote, sync)
    print(d.df.columns)
    d = d >> apply(lambda metadata, columns: metadata[columns, "id_estudo"]).df
    d = ch(d, loc, rem, local, remote, sync)
    print("Noncolinear dataset with NaNs again ------------------------------------------------------------\n", d.df, "______________________________________________________\n")

    # Remove NaNs ###############################################################################################################
    # d = d >> apply(remove_nan_rows_cols, keep=["id_estudo", target]).df
    # print("Dataset without NaNs ------------------------------------------------------------\n", d.df, "______________________________________________________\n")
