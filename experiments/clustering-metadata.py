from timeit import timeit

from matplotlib import pyplot as plt
from numpy import log, ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.manifold import smacof
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sortedness import sortedness
from sortedness.global_ import cov2dissimilarity

from hdict import _

import numpy as np
from mdscuda import MDS, mds_fit, minkowski_pairs

MAX_BLANKS_BY_ROW, MAX_BLANKS_BY_COL = 999, 3
path = "/home/davi/git/germina/data/metadata___2023-05-08-fup5afixed.csv"

blanks_by_col__tups = """0	b21_t1
0	c12f_t1
0	c12a_t1
0	c12b_t1
0	d05_t1
0	e01_t1
0	e02a_t1
0	e02c_t1
0	e03d_t1
0	e05_t1
0	bayley_1_t1
0	bayley_2_t1
0	bayley_3_t1
0	bayley_6_t1
0	bayley_7_t1
0	bayley_8_t1
0	bayley_11_t1
0	bayley_12_t1
0	bayley_13_t1
0	ibq_act_t1
0	ibq_dist_t1
0	ibq_smil_t1
0	ibq_lip_t1
0	ibq_soot_t1
0	ibq_fall_t1
0	ibq_fallr_t1
0	ibq_cudd_t1
0	ibq_sad_t1
0	ibq_voc_t1
0	ibq_sur_t1
0	ibq_neg_t1
0	ibq_reg_t1
0	ahmed_c3_t1
0	ahmed_c4_t1
0	ahmed_c5_t1
0	ahmed_c6_t1
0	ahmed_c14_t1
0	ahmed_c14_2c_t1
0	delivery_mode
0	renda_familiar_total_t0
0	antibiotic
1	a08_t1
1	b20_t1
1	e02b_t1
1	e03c_t1
1	bayley_se_score_t1
1	bayley_se_score_2_t1
1	ibq_dura_t1
1	ahmed_c7_t1
1	EBF_3m
1	infant_ethinicity
2	b04_t1
2	e04_t1
2	f04_t1
2	ibq_app_t1
2	elegib6_t0
2	maternal_ethinicity
3	b13_t1
3	e02d_t1
3	ibq_fear_t1
4	bayley_16_t1
4	bayley_17_t1
4	bayley_18_t1
4	bayley_19_t1
4	bayley_20_t1
4	bayley_21_t1
5	ahmed_c13_t1
6	droga2_t1
6	bayley_22_t1
6	bayley_23_t1
6	bayley_24_t1
6	ibq_perc_t1
8	ibq_hip_t1
13	e03d_2_t1
18	f19_t1
19	f20_t1
21	f21_t1
33	d07_t1
33	g03c_t1
34	bayley_se_score_2_t2
35	e04_t2
35	ibq_act_t2
35	ibq_dist_t2
35	ibq_dura_t2
35	ibq_smil_t2
35	ibq_hip_t2
35	ibq_lip_t2
35	ibq_soot_t2
35	ibq_fall_t2
35	ibq_fallr_t2
35	ibq_cudd_t2
35	ibq_sad_t2
35	ibq_app_t2
35	ibq_voc_t2
35	ibq_sur_t2
35	ibq_neg_t2
35	ibq_reg_t2
36	d10_t1
36	e03c_t2
36	e05_t2
38	d11_t1
38	e03d_t2
40	ibq_fear_t2
42	ibq_perc_t2
45	bayley_1_t2
45	bayley_2_t2
45	bayley_3_t2
45	bayley_6_t2
45	bayley_16_t2
45	bayley_7_t2
45	bayley_17_t2
45	bayley_18_t2
45	bayley_8_t2
45	bayley_11_t2
45	bayley_19_t2
45	bayley_12_t2
45	bayley_20_t2
45	bayley_21_t2
45	bayley_13_t2
52	e03d_2_t2
53	e03d_1_t1
54	fup_5_t2
55	bayley_22_t2
56	bayley_24_t2
57	bayley_23_t2
61	renda_t2
62	e02a_t2
62	e02b_t2
62	e02c_t2
63	e01_t2
63	e02d_t2
69	renda_total_t2
72	e04a_t2
72	e04b_t2
72	e04c_t2
72	e04d_t2
73	e04e_t2
73	e04f_t2
145	e03d_1_t2
278	e04a_t1
278	e04b_t1
278	e04c_t1
278	e04d_t1
278	e04e_t1
278	e04f_t1
314	e04b_2_t2
393	e04b_2_t1
459	fup_5a_t2
462	f04a_t2
535	ebia_tot_t2
535	ebia_2c_t2
549	f22_t1
549	f23_t1""".split("\n")
blanks_by_col = dict(tuple(reversed(tup.split("\t"))) for tup in blanks_by_col__tups)
blanks_by_col = {k: int(v) for k, v in blanks_by_col.items()}
vars = [col for col, blanks in blanks_by_col.items() if blanks <= MAX_BLANKS_BY_COL]
print(f"{vars=}")

blanks_by_id__tups = """4	249
4	203
5	212
5	116
5	168
5	263
5	204
5	309
5	321
5	379
5	407
5	540
6	53
6	87
6	200
6	247
6	243
6	20
6	54
6	207
6	97
6	181
6	153
6	104
6	219
6	149
6	315
6	206
6	232
6	285
6	314
6	310
6	303
6	347
6	327
6	336
6	342
6	337
6	402
6	437
6	366
6	395
6	378
6	469
6	391
6	507
6	413
6	430
6	512
6	523
6	527
7	142
7	183
7	25
7	37
7	106
7	190
7	240
7	167
7	253
7	230
7	159
7	59
7	174
7	122
7	213
7	216
7	136
7	164
7	157
7	182
7	173
7	165
7	312
7	259
7	330
7	331
7	297
7	324
7	320
7	417
7	341
7	332
7	432
7	328
7	333
7	370
7	358
7	385
7	456
7	355
7	364
7	484
7	412
7	404
7	472
7	384
7	457
7	519
7	374
7	450
7	396
7	394
7	513
7	497
7	481
7	474
7	415
7	556
7	495
7	496
7	467
7	426
7	453
7	460
7	475
7	473
7	560
7	518
7	509
7	502
7	517
8	135
8	185
8	120
8	50
8	67
8	95
8	171
8	126
8	19
8	141
8	38
8	224
8	139
8	78
8	44
8	88
8	257
8	211
8	64
8	255
8	148
8	102
8	283
8	269
8	223
8	313
8	236
8	271
8	197
8	218
8	250
8	266
8	237
8	318
8	251
8	306
8	335
8	340
8	345
8	338
8	397
8	416
8	359
8	350
8	448
8	492
8	480
8	377
8	503
8	421
8	478
8	435
8	425
8	544
8	461
8	489
8	442
8	499
8	488
8	479
8	436
8	536
8	516
8	483
8	538
8	522
8	543
8	555
9	35
9	65
9	13
9	18
9	96
9	16
9	111
9	81
9	287
9	295
9	258
9	260
9	291
9	372
9	387
9	360
9	369
9	380
9	403
9	477
9	422
9	459
9	515
9	535
9	505
9	551
10	66
10	98
10	147
10	138
10	305
10	547
10	521
11	75
11	163
11	28
11	29
11	180
11	124
11	214
11	189
11	409
11	452
11	440
12	195
12	30
12	158
12	40
12	133
12	123
12	79
12	225
12	118
12	193
12	349
12	351
12	406
12	414
12	465
12	447
12	485
12	533
13	21
13	82
13	45
13	71
13	34
13	172
13	47
13	186
13	242
13	143
13	192
13	83
13	63
13	42
13	93
13	125
13	169
13	152
13	272
13	293
13	205
13	241
13	256
13	231
13	296
13	325
13	282
13	304
13	356
13	326
13	334
13	352
13	433
13	443
13	408
13	389
13	431
13	383
13	410
13	399
13	451
13	398
13	444
13	434
13	520
13	553
14	24
14	15
14	146
14	23
14	131
14	41
14	57
14	51
14	109
14	150
14	92
14	160
14	177
14	46
14	62
14	127
14	55
14	121
14	184
14	261
14	31
13	89
14	217
14	60
14	69
14	58
14	166
14	161
14	140
14	117
14	108
14	154
14	70
14	80
14	179
14	105
14	155
14	175
14	170
14	292
14	298
14	288
14	300
14	210
14	226
14	194
14	252
14	244
14	262
14	322
14	281
14	267
14	311
14	279
14	284
14	276
14	268
14	299
14	289
14	275
14	329
14	317
14	343
14	375
14	354
14	471
14	357
14	362
14	405
14	401
14	371
14	363
14	386
14	494
14	427
14	455
14	441
14	390
14	393
14	411
14	449
14	458
14	429
14	439
14	504
14	510
14	454
14	508
14	419
14	500
14	464
14	506
14	493
14	539
14	534
15	74
15	7
15	11
15	6
15	73
15	33
15	110
15	56
15	209
15	85
15	220
15	248
15	274
15	234
15	302
15	420
15	376
15	428
15	418
15	462
15	528
15	524
15	501
15	537
16	5
16	39
16	52
16	235
16	228
16	151
16	112
16	137
16	188
16	308
16	264
16	265
16	323
16	339
16	392
16	368
16	438
16	470
16	552
17	2
17	32
17	115
17	100
17	201
17	316
17	400
17	361
17	424
17	382
17	466
18	10
18	99
18	86
18	27
18	208
18	76
18	215
18	178
18	36
18	145
18	191
18	222
18	294
18	246
18	278
19	128
19	176
19	43
19	114
19	49
19	103
19	130
19	229
19	290
19	381
19	557
20	22
20	119
20	14
20	129
20	90
20	202
20	101
20	61
20	162
20	254
20	221
20	280
20	238
20	233
20	239
20	270
20	353
20	344
20	367
20	546
20	446
20	548
20	529
20	559
21	68
21	132
21	286
22	26
22	17
22	187
22	91
23	1
23	113
24	77
24	277
24	486
25	227
25	463
26	514
28	8
30	94
30	491
31	199
31	301
33	3
33	12
38	134
48	365
64	144
64	156
64	307
64	445
64	549
64	532
64	550
64	558
65	48
65	84
65	273
65	319
65	388
65	511
65	482
65	476
65	526
65	541
65	554
65	525
65	542
66	107
66	531
66	498
67	468
68	373
70	196
71	346
71	348
71	423
71	487
71	530
71	545
73	490""".split("\n")
blanks_by_id = dict(tuple(reversed(tup.split("\t"))) for tup in blanks_by_id__tups)
blanks_by_id = {int(k): int(v) for k, v in blanks_by_id.items()}
ids = [id for id, blanks in blanks_by_id.items() if blanks <= MAX_BLANKS_BY_ROW]
print(f"{ids=}")

d = _.fromfile(path)
df: DataFrame = d.df
df.set_index("id_estudo", inplace=True)
# df.to_csv("/home/davi/     .csv")
df = df[vars]
df = df.loc[ids]
print("df:", df.shape)
print(df)
print()
print()
print()
nans = df.isna().sum(axis=1)
print(nans.sort_values().to_string())

print("Drop NaN rows")
print(df)
df.dropna(inplace=True)
print(df)

del df["e01_t1"]
df["antibiotic"] = df["antibiotic"] == "yes"
df["EBF_3m"] = df["EBF_3m"] == "EBF"
df["renda_familiar_total_t0"] = log(df["renda_familiar_total_t0"])
print(df.to_string())

colors = df["infant_ethinicity"]
df.pop(("maternal_ethinicity"))
std = StandardScaler()
s: ndarray = std.fit_transform(df)
pca = PCA(n_components=5)
p = pca.fit_transform(s)

explained = DataFrame(pca.components_, columns=list(df.columns))
print(explained[explained >= 0.05].transpose().dropna(how="all").to_string())

# fig = plt.figure()
# ax = fig.add_subplot(111, projection="3d")
# ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")
# plt.show()
#
# fig = plt.figure()
# ax = fig.add_subplot(111)
# ax.scatter(p[:, 0], p[:, 1], c=colors, cmap="coolwarm")
# plt.show()


# exit()

# pca = PCA(n_components=df.shape[1], random_state=0)
# p = pca.fit_transform(df)
# for i in range(1, df.shape[1]):
#     r = pwsortedness(p, p[:, :i])
#     somean, sostd = np.mean(r), np.std(r)
#     rs = sortedness(p, p[:, :i])
#     smean, sstd = np.mean(rs), np.std(rs)
#     print(i, somean, sostd, "   ", smean, sstd)

# m = cov2dissimilarity(df.transpose().cov().to_numpy())
df = DataFrame(p)
m = cdist(df,df) #cov2dissimilarity(df.cov().to_numpy())


def f() -> ndarray:
    return smacof(m, metric=True, n_components=3, random_state=0, normalized_stress=False)[0]


p = f()
print(df.shape, p.shape, m.shape)
print(sortedness(df, p))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")

# plt.show()

mds = MDS(n_dims=3)  # defines sklearn-style class


def g() -> ndarray:
    x = mds.fit(m, sqform=True)  # fits and returns embedding
    # print("mds r2: {}".format(mds.r2))  # prints R-squared value to assess quality of fit
    return x


p = g()
print(sortedness(df, p))
fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.scatter(p[:, 0], p[:, 1], p[:, 2], c=colors, cmap="coolwarm")

plt.show()

print(min([timeit(f, number=1) for _ in range(5)]))
print(min([timeit(g, number=1) for _ in range(5)]))
