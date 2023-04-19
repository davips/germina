import matplotlib.pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn.manifold import smacof

from hdict import _

cols = {'idade_crianca_meses_t1', 'idade_crianca_meses_t2',
        'educationLevelAhmedNum_t1',
        'elegib2_t0', 'elegib9_t0', 'elegib14_t0',
        'risco_total_t0', 'risco_class_t0',
        'ebia_tot_t1', 'ebia_2c_t1',
        # 'ebia_tot_t2', 'ebia_2c_t2',
        'epds_tot_t1', 'epds_2c_t1', 'epds_tot_t2', 'epds_2c_t2',
        'pss_tot_t1', 'mspss_tot_t1', 'pss_2c_t1', 'pss_tot_t2', 'pss_2c_t2',
        'gad_tot_t1', 'gad_2c_t1', 'gad_tot_t2', 'gad_2c_t2',
        'psi_pd_t1', 'psi_pcdi_t1', 'psi_dc_t1', 'psi_tot_t1',
        # 'final_8_t1',
        'bisq_3_mins_t1', 'bisq_4_mins_t1', 'bisq_9_mins_t1', 'bisq_sleep_prob_t1', 'bisq_sleep_prob_t2',
        # 'final_10_t1', 'final_10_t2',
        'chaos_tot_t1',
        "bayley_3_t1",
        # 'bayley_1_t1', 'bayley_2_t1', 'bayley_6_t1', 'bayley_16_t1', 'bayley_7_t1', 'bayley_17_t1', 'bayley_18_t1', 'bayley_8_t1', 'bayley_11_t1', 'bayley_19_t1', 'bayley_12_t1', 'bayley_20_t1', 'bayley_21_t1', 'bayley_13_t1', 'bayley_22_t1', 'bayley_23_t1', 'bayley_24_t1', 'bayley_1_t2', 'bayley_2_t2', 'bayley_3_t2', 'bayley_6_t2', 'bayley_16_t2', 'bayley_7_t2', 'bayley_17_t2', 'bayley_18_t2', 'bayley_8_t2', 'bayley_11_t2', 'bayley_19_t2', 'bayley_12_t2', 'bayley_20_t2',
        # 'bayley_21_t2', 'bayley_13_t2', 'bayley_22_t2', 'bayley_23_t2', 'bayley_24_t2',
        # 'ibq_sur_t1', 'ibq_neg_t1', 'ibq_reg_t1', 'ibq_sur_t2', 'ibq_neg_t2', 'ibq_reg_t2',
        'Delta', 'Theta', 'HighAlpha', 'Beta', 'Gamma', 'Number segments',
        '2Hz pre-post wavelet change', '5Hz pre-post wavelet change', '12Hz pre-post wavelet change', '20Hz pre-post wavelet change', '30Hz pre-post wavelet change',
        'N1 peak amplitude', 'VEP Number segments',
        'chao1',
        'diversity_shannon', 'dominance_simpson',
        'k__Bacteria.p__Proteobacteria.c__Gammaproteobacteria.o__Enterobacterales.f__Enterobacteriaceae.g__Escherichia.s__Escherichia_coli', 'k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Bacteroidaceae.g__Bacteroides.s__Bacteroides_vulgatus', 'k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Bifidobacteriales.f__Bifidobacteriaceae.g__Bifidobacterium.s__Bifidobacterium_bifidum',
        'k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Bifidobacteriales.f__Bifidobacteriaceae.g__Bifidobacterium.s__Bifidobacterium_longum', 'k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Clostridiaceae.g__Clostridium.s__Clostridium_neonatale', 'k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Bacteroidaceae.g__Bacteroides.s__Bacteroides_fragilis', 'k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Blautia.s__Ruminococcus_gnavus',
        's__Ruthenibacterium_lactatiformans', 's__Bacteroides_uniformis', 's__Flavonifractor_plautii', 's__Campylobacter_concisus', 's__Morganella_morganii', 's__Klebsiella_oxytoca', 'g__Bifidobacterium', 'g__Lactobacillus', 'g__Bacteroides', 'g__Streptococcus', 'g__Clostridium', 'g__Blautia'}

drop = set("elegib14_t0,bisq_3_mins_t1,bisq_4_mins_t1,g__Bacteroides,g__Bifidobacterium,bsq_9_mins_t1,k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Bacteroidaceae.g__Bacteroides.s__Bacteroides_vulgatus,psi_tot_t1,bayley_24_t2,k__Bacteria.p__Bacteroidetes.c__Bacteroidia.o__Bacteroidales.f__Bacteroidaceae.g__Bacteroides.s__Bacteroides_fragilis,k__Bacteria.p__Proteobacteria.c__Gammaproteobacteria.o__Enterobacterales.f__Enterobacteriaceae.g__Escherichia.s__Escherichia_coli,"
           "VEP Number segments,pss_tot_t2,k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Clostridiaceae.g__Clostridium.s__Clostridium_neonatale,g__Clostridium,k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Bifidobacteriales.f__Bifidobacteriaceae.g__Bifidobacterium.s__Bifidobacterium_bifidum,psi_pd_t1,"
           "bisq_9_mins_t1,bayley_8_t2,k__Bacteria.p__Firmicutes.c__Clostridia.o__Clostridiales.f__Lachnospiraceae.g__Blautia.s__Ruminococcus_gnavus,k__Bacteria.p__Actinobacteria.c__Actinobacteria.o__Bifidobacteriales.f__Bifidobacteriaceae.g__Bifidobacterium.s__Bifidobacterium_longum,"
           "g__Blautia,bayley_13_t2,bayley_3_t2,bayley_24_t1,bayley_22_t2,bayley_1_t1,bayley_13_t1,bayley_8_t1,".split(","))
cols = cols.difference(drop)
path = "/home/davi/git/germina/data/"
data = {
    "socio": _.fromfile(path + "grover230303---2023-04-06.csv"),
    "eeg": _.fromfile(path + "eeg/all.csv"),
    "mbiome_alpha": _.fromfile(path + "microbiome/alpha_diversity_microbiome---2023-04-06.csv"),
    "mbiome_abun": _.fromfile(path + "microbiome/most_abundant_species---email1.csv"),
    "mbiome_vars": _.fromfile(path + "microbiome/variaveis_microbioma_integracao_dados---email2.csv")
}
dfs = [d.df.set_index("id_estudo") for d in data.values()]
df: DataFrame = dfs[0].join(dfs[1:], how="outer")[list(cols)]
print("shape", df.shape)

# Remove rows with lots of NaNs.
max_nans = 320
bestrows_mask = df.notna().sum(axis=1) < max_nans
df = df[bestrows_mask]
print(df.shape)

colors = df.pop("bayley_3_t1")

m = df.transpose().cov()
# m = m[m > -1][m < 1]
# print(m.count().sort_values().to_string())

fig = plt.figure()

p: ndarray = smacof(m.fillna(0), metric=False, n_components=2, random_state=0)[0]
ax = fig.add_subplot(111)
ax.scatter(p[:, 0], p[:, 1], c=colors, cmap="coolwarm")
# p: ndarray = smacof(m.fillna(0), metric=False, n_components=3, random_state=0)[0]
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")

plt.show()

# ax.set_xlim([0.099, 0.8])
# ax.set_ylim([0.9975, 1.0001])

# # ax.set_title('Loss curve', fontsize=15)
# plt.rcParams["font.size"] = 35
# for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
#     item.set_fontsize(plt.rcParams["font.size"])
# for (ylabel, data), (style, width, color) in zip(list(d.items())[1:], [
#     ("dotted", 1.5, "blue"),
#     ("dotted", 3, "orange"),
#     ("dotted", 3, "black"),
#     ("-.", 3, "red"),
#     ("dashed", 3, "purple"),
#     ("dashed", 1.5, "brown"),
# ]):
#     print("\n" + ylabel)
#     df.plot.line(ax=ax, y=[ylabel], linestyle=style, lw=width, color=color, logy=False, logx=False, fontsize=plt.rcParams["font.size"])
#
# plt.grid()
#
# plt.legend(loc=3)
# plt.ylabel("")
# plt.subplots_adjust(left=0.07, bottom=0.14, right=0.995, top=0.99)
# arq = '/home/davi/git/articles/sortedness/images/translation.pgf'
# plt.savefig(arq, bbox_inches='tight')
# with open(arq, "r") as f:
#     txt = f.read().replace("sffamily", "rmfamily")
# with open(arq, "w") as f:
#     f.write(txt)
# # plt.show()
