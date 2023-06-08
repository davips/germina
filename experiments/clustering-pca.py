from pprint import pprint

import matplotlib.pyplot as plt
import numpy as np
from numpy import ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.cluster import AgglomerativeClustering as HC
from sklearn.decomposition import PCA
from sklearn.manifold import smacof

from experimentssortedness.temporary import sortedness, pwsortedness, cov2dissimilarity, global_pwsortedness
from hdict import _

# elegib2_t0: Idade da mãe; elegib4_t0: Em qual cidade?; elegib14_t0: Qual o sexo do seu bebê?
cols = """psi_pcdi_t1
psi_tot_t1
gad_2c_t1
ibq_neg_t1
bayley_3_t1
elegib2_t0
bayley_2_t1
educationLevelAhmedNum_t1
risco_total_t0
bayley_11_t1
bayley_7_t1
ibq_sur_t1
mspss_tot_t1
psi_pd_t1
bayley_6_t1
epds_2c_t1
elegib14_t0
psi_dc_t1
idade_crianca_meses_t1
ebia_2c_t1
epds_tot_t1
ebia_tot_t1
bayley_8_t1
bayley_12_t1
gad_tot_t1
risco_class_t0
ibq_reg_t1
bayley_1_t1
pss_2c_t1
pss_tot_t1
bayley_13_t1
bisq_3_mins_t1
bisq_9_mins_t1
bisq_4_mins_t1
chaos_tot_t1
elegib9_t0
bayley_20_t1
bayley_17_t1
bisq_sleep_prob_t1
bayley_19_t1
bayley_21_t1
bayley_18_t1
bayley_16_t1
bayley_24_t1
bayley_23_t1
bayley_22_t1""".split("\n")
# categorical deleted: final_8_t1 final_10_t1

rows = map(int, "1	2	5	6	7	8	10	11	12	13	14	15	16	17	18	19	20	21	22	23	24	25	26	27	28	29	30	31	32	33	34	35	36	37	38	39	40	43	44	45	46	47	48	49	51	52	53	54	55	56	57	58	59	60	61	62	63	64	65	66	67	68	69	70	71	73	74	75	76	78	79	80	81	82	83	84	85	86	87	88	89	90	92	93	94	95	96	97	98	100	101	102	103	104	105	106	107	108	109	110	111	112	114	115	116	117	118	119	121	122	123	124	125	126	127	128	129	130	131	132	133	134	" \
                "135	136	137	138	139	140	141	142	143	144	145	146	147	148	149	150	151	152	153	154	155	156	157	158	159	160	161	162	163	164	165	166	167	168	169	170	171	172	173	174	175	176	177	178	179	180	181	182	183	184	185	186	187	188	189	190	191	192	193	194	195	196	197	197	199	200	201	202	203	204	205	206	207	208	209	210	211	212	213	214	215	216	217	218	219	220	221	222	223	224	225	226	227	228	229	230	231	232	233	234	235	236	237	238	239	240	241	242	243	244	246	247	248	249	250	251	252	253	254	255	256	257	" \
                "258	259	260	261	262	263	264	265	266	267	268	269	270	271	272	273	274	275	276	277	278	279	280	281	282	283	284	285	286	287	288	289	290	291	292	293	294	295	296	297	298	299	300	301	302	303	304	305	306	307	308	309	310	311	312	313	314	315	316	317	318	319	320	321	322	323	324	325	326	327	328	329	330	331	332	333	334	335	336	337	338	339	340	341	342	343	344	345	346	347	348	349	350	351	352	353	354	355	356	357	358	359	360	361	362	363	364	365	366	367	368	369	370	371	372	373	374	375	376	377	378	379	" \
                "380	381	383	384	385	386	387	388	389	390	391	392	393	394	395	396	397	398	399	400	401	402	403	404	405	406	407	408	409	410	411	412	413	414	415	416	417	419	420	421	422	423	424	425	426	427	428	429	430	431	432	433	434	435	436	437	438	439	440	441	442	443	444	445	446	447	448	449	450	451	452	453	454	455	456	457	458	459	460	461	462	463	464	465	466	467	468	469	471	472	473	474	475	476	477	478	479	480	481	482	483	484	485	486	487	488	489	490	491	492	493	494	495	496	497	499	500	501	502	503	504	505	" \
                "506	507	509	510	511	512	513	514	515	516	517	518	519	520	522	523	524	525	526	527	528	529	530	531	532	534	535	536	537	538	539	540	541	542	543	544	545	546	547	548	549	550	551	552	553	554	555	556	557	558	559	560".split("\t"))
path = "/home/davi/git/germina/data/"
data = {
    "socio": _.fromfile(path + "grover230303---2023-04-06.csv"),
    "eeg": _.fromfile(path + "eeg/all.csv"),
    "mbiome_alpha": _.fromfile(path + "microbiome/alpha_diversity_microbiome---2023-04-06.csv"),
    "mbiome_abun": _.fromfile(path + "microbiome/most_abundant_species---email1.csv"),
    "mbiome_vars": _.fromfile(path + "microbiome/variaveis_microbioma_integracao_dados---email2.csv")
}
dfs = [d.df.set_index("id_estudo") for d in data.values()]
df: DataFrame = dfs[0].join(dfs[1:], how="outer")
# df.to_csv("/home/davi/socio-eeg-alpha-abundance-mbiomevars.csv")
df = df[list(cols)]
df = df.loc[rows]
print("df:", df.shape)

colors = df.pop("elegib14_t0")
pca = PCA(n_components=4)
p = pca.fit_transform(df)
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")
plt.show()

explained = DataFrame(pca.components_, columns=list(df.columns))
print(explained[explained >= 0.02].transpose().to_string())
pca = PCA(n_components=df.shape[1], random_state=0)
p = pca.fit_transform(df)
for i in range(1, df.shape[1]):
    r = pwsortedness(p, p[:, :i])
    somean, sostd = np.mean(r), np.std(r)
    rs = sortedness(p, p[:, :i])
    smean, sstd = np.mean(rs), np.std(rs)
    print(i, somean, sostd, "   ", smean, sstd)
