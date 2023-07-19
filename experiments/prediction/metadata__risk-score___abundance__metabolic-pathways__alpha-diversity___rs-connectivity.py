from pprint import pprint

from matplotlib import pyplot as plt
from pandas import Series, DataFrame
from sklearn.dummy import DummyClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold, cross_val_score, cross_val_predict
import numpy as np
import pandas as pd
from numpy import quantile, mean, std
from sklearn.ensemble import RandomForestClassifier as RFc
from sklearn.ensemble import RandomForestRegressor as RFr

from germina.config import local_cache_uri
from germina.data import clean
from hdict import _, apply, cache
from shelchemy import sopen

"""
'ebia_tot_t2', 'ebia_2c_t2', 'bayley_1_t2', 'bayley_2_t2',
       'bayley_3_t2', 'bayley_6_t2', 'bayley_16_t2', 'bayley_7_t2',
       'bayley_17_t2', 'bayley_18_t2', 'bayley_8_t2', 'bayley_11_t2',
       'bayley_19_t2', 'bayley_12_t2', 'bayley_20_t2', 'bayley_21_t2',
       'bayley_13_t2', 'bayley_22_t2', 'bayley_23_t2', 'bayley_24_t2',
       'risco_total_t0'
"""
files = [
    ("data_microbiome___2023-06-18___alpha_diversity_n525.csv", None),
    ("data_microbiome___2023-06-20___vias_metabolicas_3_meses_n525.csv", None),
    ("data_microbiome___2023-06-18___especies_3_meses_n525.csv", None),
    ("data_eeg___2023-06-20___T1_RS_average_dwPLI_withEEGCovariates.csv", None),
    ("data_eeg___2023-06-20___T2_RS_average_dwPLI_withEEGCovariates.csv", None),
    ("metadata___2023-06-18.csv", ["id_estudo", "ibq_reg_t1", "ibq_reg_t2"]),  # "risco_class": colocar de volta depois de conferido por Pedro
    # ("metadata___2023-06-18.csv", None)
]
targets = ["ibq_reg_t1", "ibq_reg_t2", "ibq_reg_t2-ibq_reg_t1"]
d = clean(targets, "data/", files, [local_cache_uri], mds_on_first=False)

# daf: DataFrame = d.df["idade_crianca_meses_t1"]
# th = 3.6
# lo, hi = daf[daf <= th], daf[daf > th]
# print(lo.count(), hi.count())
# # mn, mx = quantile(out, [1 / 3, 2 / 3])
#
#
# outcome: DataFrame = d.df["ibq_reg_t2"]
# print(list(outcome.sort_values()))
#
# tho = outcome.median()
# a, b = daf[outcome <= tho], daf[outcome > tho]
#
# # plt.hist([a, b], bins=12)
#
#
# outcome2: DataFrame = d.df["bayley_3_t2"]
# outcome2.hist(bins=22)
# plt.show()


pprint([col[:80] for col in d.raw_df.columns])

cm = "coolwarm"
for target in targets:
    if "-" in target:
        a, b = target.split("-")
        labels = d.targets[a] - d.targets[b]
    else:
        labels = d.targets[target]
        if target.startswith("ibq_reg_t"):
            # qmn, qmx = quantile(labels, [1 / 4, 3 / 4])
            qmn, qmx = quantile(labels, [1 / 2, 1 / 2])
            menores = labels[labels <= qmn].index
            maiores = labels[labels >= qmx].index
            pd.options.mode.chained_assignment = None
            labels.loc[menores] = -1
            labels.loc[maiores] = 1
            labels.loc[list(set(labels.index).difference(maiores, menores))] = 0

    X = d.raw_df
    y = labels  # np.where(labels == 2, 1, labels)
    cv = KFold(n_splits=5, random_state=0, shuffle=True)
    with sopen(local_cache_uri) as local:
        d >>= (
                apply(RFc, n_estimators=1000, random_state=0, n_jobs=-1).rfc
                >> apply(cross_val_score, _.rfc, X=X, y=y, cv=cv, n_jobs=-1).scoresc
                >> apply(DummyClassifier).dc
                >> apply(cross_val_score, _.dc, X=X, y=y, cv=cv, n_jobs=-1).baseline
                >> apply(RFr, n_estimators=1000, random_state=0, n_jobs=-1).rfr
                >> apply(cross_val_score, _.rfr, X=_.raw_df, y=labels, cv=cv, scoring="r2", n_jobs=-1).scoresr
                >> cache(local)
        )
        print(confusion_matrix(y, cross_val_predict(d.rfc, X, y)))
        print(
            f"{target.ljust(20)}\t"
            f"baseline:\t{mean(d.baseline):.2f} ± {std(d.baseline):.2f}\t\t"
            f"RFcla:\t{mean(d.scoresc):.2f} ± {std(d.scoresc):.2f}\t\t"
            f"RFreg:\t{mean(d.scoresr):.2f} ± {std(d.scoresr):.2f}\t\t"
        )
        print(",".join(map(str, d.baseline)))
        print(",".join(map(str, d.scoresc)))
        # print(",".join(d.scoresr))
    print()

""" EEG + Microbiome + SocioDemographic
risco_class         	baseline:	0.80 ± 0.03		RFcla:	0.82 ± 0.03		RFreg:	0.76 ± 0.13		
0.8048780487804879,0.8292682926829268,0.7804878048780488,0.8048780487804879,0.8048780487804879,0.8292682926829268,0.7804878048780488,0.8536585365853658,0.8048780487804879,0.7560975609756098
0.8292682926829268,0.8536585365853658,0.7804878048780488,0.8292682926829268,0.8048780487804879,0.8536585365853658,0.8292682926829268,0.8780487804878049,0.7804878048780488,0.8048780487804879

ibq_reg_t1          	baseline:	0.50 ± 0.07		RFcla:	0.56 ± 0.06		RFreg:	0.71 ± 0.06		
0.43902439024390244,0.5609756097560976,0.43902439024390244,0.5853658536585366,0.43902439024390244,0.4146341463414634,0.4146341463414634,0.5121951219512195,0.5853658536585366,0.5853658536585366
0.4878048780487805,0.6585365853658537,0.4634146341463415,0.5609756097560976,0.5365853658536586,0.5609756097560976,0.5853658536585366,0.6097560975609756,0.4878048780487805,0.6097560975609756

ibq_reg_t2          	baseline:	0.50 ± 0.07		RFcla:	0.53 ± 0.07		RFreg:	0.66 ± 0.09		
0.3902439024390244,0.5365853658536586,0.5365853658536586,0.6341463414634146,0.5365853658536586,0.43902439024390244,0.5121951219512195,0.3902439024390244,0.5365853658536586,0.4634146341463415
0.36585365853658536,0.5365853658536586,0.6097560975609756,0.5853658536585366,0.5365853658536586,0.6097560975609756,0.5853658536585366,0.4634146341463415,0.5609756097560976,0.4634146341463415
"""

""" Microbiome
risco_class         	baseline:	0.80 ± 0.04		RFcla:	0.80 ± 0.05		RFreg:	0.69 ± 0.12		
0.7755102040816326,0.7755102040816326,0.8367346938775511,0.7551020408163265,0.875,0.8333333333333334,0.7916666666666666,0.8541666666666666,0.7291666666666666,0.7708333333333334
0.7959183673469388,0.7346938775510204,0.8571428571428571,0.7551020408163265,0.8541666666666666,0.8333333333333334,0.7916666666666666,0.8541666666666666,0.7291666666666666,0.7708333333333334

ibq_reg_t1          	baseline:	0.50 ± 0.08		RFcla:	0.43 ± 0.06		RFreg:	-0.12 ± 0.06		
0.40816326530612246,0.5102040816326531,0.6326530612244898,0.5510204081632653,0.3333333333333333,0.5625,0.4791666666666667,0.4791666666666667,0.4583333333333333,0.5625
0.3877551020408163,0.42857142857142855,0.5102040816326531,0.46938775510204084,0.2916666666666667,0.4375,0.4791666666666667,0.4583333333333333,0.3541666666666667,0.4791666666666667

ibq_reg_t2          	baseline:	0.50 ± 0.07		RFcla:	0.47 ± 0.07		RFreg:	-0.12 ± 0.08		
0.5714285714285714,0.5510204081632653,0.4489795918367347,0.6122448979591837,0.4791666666666667,0.4166666666666667,0.5625,0.4583333333333333,0.4583333333333333,0.4166666666666667
0.5510204081632653,0.5510204081632653,0.40816326530612246,0.5918367346938775,0.4166666666666667,0.375,0.4583333333333333,0.4375,0.5,0.3958333333333333

ibq_reg_t2-ibq_reg_t1	baseline:	0.50 ± 0.09		RFcla:	0.46 ± 0.07		RFreg:	-0.17 ± 0.12		
0.4897959183673469,0.46938775510204084,0.5510204081632653,0.4897959183673469,0.4166666666666667,0.3958333333333333,0.6458333333333334,0.6458333333333334,0.3958333333333333,0.5416666666666666
0.46938775510204084,0.4489795918367347,0.5306122448979592,0.4489795918367347,0.4375,0.2916666666666667,0.5416666666666666,0.5416666666666666,0.3958333333333333,0.4791666666666667
"""


"""
# "elegib6_t0",  # mais adultos com renda?
# "a10_t1",  # ordem desta criança entre os irmãos
# "b20_t1",  # Quantas pessoas adultas (maiores de 18 anos de idade), incluindo você, residem no seu domicílio?
# "b21_t1",  # Quantas crianças e adolescentes (menores de 18 anos de idade), residem no seu domicílio?
# "d04_t1",  # Tipo de parto 1	Normal 2	Parto normal com fórceps ou extrator a vácuo sem 3	Parto cesáreo

# "c12a_t1",  # Você teve diabetes gestacional?
# "c12b_t1",  # Você teve pré-eclâmpsia ou eclâmpsia?
# "c12f_t1",  # Depressão durante a gestação ou no pós-parto
# "d05_t1",  # Seu bebê nasceu com quantas semanas de gestação?
# "d07_t1",  # Qual o peso do seu bebê no nascimento? (solicitar documento de alta hospitalar para registrar o dado correto)"
# "d10_t1",  # Qual foi o APGAR de 1º minuto?
# "d11_t1",  # Qual foi o APGAR de 5º minuto?
# "e01_t1",  # Quanto tempo após o nascimento o (a) [nome_bebe] foi colocado (a) ao peito?Imediatamente 000.Se menos de 1 hora 00.Se menos de 24hs, coloque as HORAS: __:__De outra forma, coloque os DIAS: ___"
# "e02a_t1",  # Chá, água ou glicose?  0	Não 1	Sim
# "e02b_t1",  # Bico ou chupeta?   0	Não 1	Sim
# "e02c_t1",  # Mamadeira de leite materno ou fórmula? 0	Não 1	Sim
# "e02d_t1",  # Leite materno de copo ou colher?   0	Não 1	Sim
# "e03c_t1",  # : Até quando ele (a) mamou exclusivamente ao peito, sem tomar água, chá ou outros líquidos? Menos de 1 mês 000Meses ___
# "e03d_t1",  # : Ainda mama?
# "e03d_1_t1",  # SE SIM, QUANTAS VEZES AO DIA
# "e03d_2_t1",  # Até quando ele (a) mamou ao peito? (número de dias que o bebê estava quando mamou pela última vez)"
# "e04_t1",  # Houve ingestão de líquidos? 1	Sim 0	Não
# "e04b_t1",  # Fórmula Infantil (NAN, APTAMIL, NESTOGENO, MILUPA, ENFAMIL, outras)"  1	Sim 0	Não 99	Não sei
# "e04b_2_t1",  # Se SIM, a fórmula foi oferecida com açúcar?  1	Sim 0	Não 99	Não sei
# "e04c_t1",  # Leite de Vaca (em todas as formas- NINHO, PAULISTA,PIRACANJUBA, ITALAC, TIROL, outros)"   1	Sim 0	Não 99	Não sei
# "e04d_t1",  # Composto Lácteo (NINHO FASES-1+,2+, 3+, MILNUTRI, NESLAC, ENFAGROW)    1	Sim 0	Não 99	Não sei
# "e04e_t1",  # Foi acrescentado à Fórmula, Leite de Vaca e ou Composto Lácteo: MUCILON, NESTON, FARINHA LÁCTEA, ARROZINA, MAIZENA 1	Sim 0	Não 99	Não sei
# "e04f_t1",  # Bebidas tipo iogurtes - bebidas lácteas (DANONE, NESTLÉ, Yakult, outros)   1	Sim 0	Não 99	Não sei
# "e05_t1",  # Houve ingestão de alimentos?    1	Sim 0	Não e05a_t1: Iogurte (que se come de colher) (Danoninho, Chambinho, Activia, outros)    1	Sim 0	Não 99	Não sei
# "f04_t1",  # Seu bebê já teve alguma infecção, como infecção de ouvido, garganta, pulmão, etc?"
# "f19_t1",  # Peso** Em qual unidade de medida será coletado? Quantas casas decimais padronizaremos?
# "f20_t1",  # Comprimento ** Em qual unidade de medida será coletado? Quantas casas decimais padronizaremos?
# "f21_t1",  # Circunferência cefálica** Em qual unidade de medida será coletado? Quantas casas decimais padronizaremos?
# "f22_t1",  # Circunferência torácica** Em qual unidade de medida será coletado? Quantas casas decimais padronizaremos?
# "f23_t1",  # Circunferência abdominal** Em qual unidade de medida será coletado? Quantas casas decimais padronizaremos?
# "bisq_3_mins_t1",  # Quanto tempo seu(sua) filho(a) passa dormindo durante a NOITE (entre 7 da noite e 7 da manhã)? (em minutos)
# "bisq_4_mins_t1",  # Quanto tempo seu(sua) filho(a) passa dormindo durante o DIA (entre 7 da manhã e 7 da noite)? (em minutos)
# "bisq_9_mins_t1",  # A que horas normalmente seu filho(a) adormece à noite? (em minutos e centrada nas 20 horas, e.g., 19h30 -> -30, 21h30 -> 90
# "bisq_sleep_prob_t1",  # BISQ: any sleep problem according to Sadeh 2004
# "mspss_tot_t1",  # Multidimensional Perceived Social Support Scale Total Score
# "chaos_tot_t1",  # Confusion, hubbub, and order scale (CHAOS) 'Total Score'
# "pss_tot_t1",  # PSS Total Score
# "pss_2c_t1",  # PSS Classification   0	PSS Total Score <= 26   1	EPDS Total Score > 26
# "gad_tot_t1",  # Generalized Anxiety Disorder-7 (GAD-7) Total Score
# "gad_2c_t1",  # GAD-7 Classification according to Spitzer et al. 2006
# "psi_pd_t1",  # PSI-IV Short Form Domain 'Parental Distress'
# "psi_pcdi_t1",  # PSI-IV Short Form Domain 'Parent-child Dysfunctional Interaction'
# "psi_dc_t1",  # PSI-IV Short Form Domain 'Difficult Child'
# "psi_tot_t1",  # PSI-IV Short Form Domain 'Total Stress'

# "bayley_3_t1",  # Cognitivo - Pontuação Composta
# "bayley_8_t1",  # Linguagem - Pontuação Composta
# "bayley_13_t1",  # Motora - Pontuação Composta
# "bayley_24_t1",  # Socioemocional - Pontuação Composta

# # "bayley_1_t1",  # Cognitivo - Pontuação Bruta Total
# # "bayley_2_t1",  # Cognitivo - Pontuação Escalonada
# # "bayley_6_t1",  # Comunicação Receptiva - Pontuação Bruta Total
# # "bayley_16_t1",  # Comunicação Receptiva - Pontuação Escalonada
# # "bayley_7_t1",  # Comunicação Expressiva - Pontuação Bruta Total
# # "bayley_17_t1",  # Comunicação Expressiva - Pontuação Escalonada
# # "bayley_18_t1",  # Linguagem - Soma
# # "bayley_11_t1",  # Motricidade Fina - Pontuação Bruta Total
# # "bayley_19_t1",  # Motricidade Fina - Pontuação Escalonada
# # "bayley_12_t1",  # Motricidade Grossa - Pontuação Bruta Total
# # "bayley_20_t1",  # Motricidade Grossa - Pontuação Escalonada
# # "bayley_21_t1",  # Motora - Soma
# # "bayley_22_t1",  # Socioemocional - Pontuação Bruta Total
# # "bayley_23_t1",  # Socioemocional - Pontuação Escalonada
#
# # "elegib26_t0", "elegib27_t0", # etnia mãe
# # "elegib5_t0",  # escolaridade
# # "elegib31_t0",  # mais filhos?
# # "elegib28_t0",  # bolsa família?
# # "elegib30_t0",  # companheiro?
# # "elegib7_t0",  # soma das rendas
# # "elegib11_t0",  # idade em dias
# # "elegib12_t0",  # duração semanas de gestação
# # "elegib13_t0",  # peso ao nascer
# # "b04_2c_t1",  # Maternal skin color white
# # "b06_t1",# Qual sua situação de trabalho atual
# # "b15_t1", # Qual sua situação de trabalho atual do pai
# # "b08_t1",# Qual o seu estado civil?
# # "b08_2c_t1",# Marital status: married or living with partner
# # "b09_t1",  # Qual sua relação com o pai de seu filho(a)? 'Presença paterna'
# # "b12_t1",  # Idade do pai
# # "b14_t1",  # Escolaridade do pai
# # "b23_t1",  # Você recebe algum benefício de Programas Sociais do Governo?
# # "b23a_t1",  # Bolsa Família
# # "b23b_t1",  # Programa Jovem Adolescente/Agente Jovem
# # "b23c_t1",  # Programa de Erradicação do Trabalho Infantil (PETI)
# # "b23d_t1",  # Benefício de Prestação Continuada de Assistência Social (BPC)
# # "b23e_t1",  # Viva Leite ou Leve Leite
# # "b23f_t1",  # Renda Cidadã
# # "b23g_t1",  # Renda Mínima
# # "c1a_t1",  # Diabetes
# # "c1b_t1",  # Pressão alta ou hipertensão
# # "c1c_t1",  # Depressão
# # "c1d_t1",  # Ansiedade
# # "c1e_t1",  # Asma, bronquite crônica ou doença pulmonar obstrutiva
# # "c1f_t1",  # Problema cardíaco ou do coração
# # "c1g_t1",  # Qualquer anormalidade congênita ou doença genética
# # "c1h_t1",  # Alguma doença auto-imune, como lupus, tireoidite, etc?
# # "any_disease_2c_t1",  # Reported any disease (c1a c1b c1e c1f c1g c1h)
# # "c2_t1",  # Atualmente você faz uso de medicação de uso contínuo?
# # "c2desc_t1",  # Se sim, qual
# # "c3_t1",  # Quantos filhos biológicos você tem?
# # "c4_t1",  # Quantas gestações você já teve?
# # "c5_t1",  # Houve algum aborto?
# # "c6_t1",  # Houve algum nascido morto?
# # "c7_t1",  # Quantos filhos adotivos você tem?
# # "c8_t1",  # Qual sua altura? (cm)
# # "c9_t1",  # Qual era o seu peso antes de engravidar? (kg)
# # "c10_t1",  # Qual era o seu peso quando o bebê nasceu?
# # "c11_t1",  # Qual é o seu peso atual?
# # "c12c_t1",  # Você teve alguma outra doença relacionada à gestação?
# # "c12d_t1",  # Você teve infecções urinárias durante a gestação?
# # "c12ddesc_t1",  # Se sim, quantas
# # "c12e_t1",  # Alguma doença sexualmente transmissível?
# # "c12edesc_t1",  # Se sim, qual?
# # "c12g_t1",  # Ansiedade
# # "c13_t1",  # Durante a gestação, você fez uso de quais medicações e com que frequência.
# # "d03_t1",  # Onde o seu bebê nasceu? 1	Hospital 2	Na minha casa 3	Outro
# # "d06_t1",  # Teve complicações no parto?
# # "d08_t1",  # Qual o comprimento do seu bebê no nascimento? (solicitar documento de alta hospitalar para registrar o dado correto)"
# # "d09_t1",  # Seu bebê teve alta junto de você?
# # "d12_t1",  # Seu filho(a) teve febre?
# # "d12a_t1",  # Foi medicado?
# # "d13_t1",  # Seu filho (a) teve cólica persistente (por mais de 3 dias)?
# # "d13a_t1",  # Foi medicado?
# # "d13c_t1",  # : Você fez algum cuidado caseiro para manejar a cólica?
# # "d14_t1",  # : Número de visitas a emergência/pronto socorro?
# # "d15_t1",  # : Algum problema de saúde ou acidente (convulsões/ acidentes)?
# # "d16_t1",  # : Ele teve alguma internação após o parto(ficou mais que 24h no hospital)?
# # "d18___1_t1",  # Quem são as pessoas que cuidaram do seu filho no último mês? (choice=Mãe)
# # "d18___2_t1",  # Quem são as pessoas que cuidaram do seu filho no último mês? pai
# # "d18___3_t1",  # Quem são as pessoas que cuidaram do seu filho no último mês? avós
# # "d18___4_t1",  # Quem são as pessoas que cuidaram do seu filho no último mês? familiar
# # "d18___5_t1",  # Quem são as pessoas que cuidaram do seu filho no último mês? nao e familiar
# # "d18___99_t1",  # Quem são as pessoas que cuidaram do seu filho no último mês?(choice=Não sei)"
# # "d19_t1",  # Quem tem sido o principal cuidador do bebe? 1	Mae 2	Pai 3	Avós maternos/avós paternos 4	Outro familiar  5	Uma pessoa que nao e familiar
# # "d20_t1",#  Há um cuidador que deixou de trabalhar para ficar com o bebe?
# # "d21_t1",#  Seu filho (a) frequenta creche ou fica na casa de outra pessoa (como vizinha) que recebe para isso?"
# # "d22_t1",#  Onde mora o pai do bebê hoje?
# # "d22a_t1",#  Caso não more na mesma casa. Quando ele vê (visita) a criança?
# # "d22b_t1",#  Tem alguém na casa que desempenhe papel de ''pai'' para seu(a) filho(a)?
# # "d23_t1",#  O Seu filho caiu e se machucou?
# # "d23a_t1",#  Caso sim, quantas vezes
# # "d24_t1",#  O seu filho se cortou?  0	Nao 1	Sim
# # "d24a_t1",#  Caso sim, quantas vezes    1	1;vez   2	2;vezes
# # "d25_t1",#  O Seu filho se queimou? 0	Nao 1	Sim
# # "d25a_t1",#  Caso sim, quantas vezes    1	1;vez   2	2;vezes
# # "d26_t1",#  O seu filho teve outro tipo de acidente?    0	Nao 1	Sim
# # "d26a_t1",#  Caso sim, quantas vezes    1	1;vez   2	2;vezes
# # "sleep_t1",  # Nos últimos 7 dias, como você avaliaria a qualidade do seu sono?
# # "e03a_t1",  # O (a) [nome_bebe] foi amamentado (a) ao peito desde ontem de manhã até hoje de manhã? 1	Sim 0	Não 99	Não sei
# # "e03b_t1",  # O (a) [nome_bebe] recebeu algum líquido através de mamadeira com bico desde ontem de manhã até hoje de manhã?"    1	Sim 0	Não 99	Não sei
# # "04a_t1",  # Água   1	Sim 0	Não 99	Não sei
# # "e04b_1_t1",  # Se SIM, quantas vezes seu filho (a) ingeriu a fórmula?
# # "e04d_1_t1",  # Se SIM, quantas vezes seu filho (a) ingeriu o composto lácteo?
# # "e04d_2_t1",  # Se SIM, o composto lácteo foi oferecido com açúcar?  1	Sim 0	Não 99	Não sei
# # "e04g_t1",  # Achocolatados (Nescau, Toddy, Toddynho, etc)   1	Sim 0	Não 99	Não sei
# # "e04h_t1",  # Sucos de frutas (natural)  1	Sim 0	Não 99	Não sei
# # "e04i_t1",  # Sucos de caixa (néctares), sucos em pó, água de coco em caixinha, xaropes de guaraná/groselha, suco de fruta com adição de açúcar  1	Sim 0	Não 99	Não sei
# # "e04j_t1",  # Refrigerantes  1	Sim 0	Não 99	Não sei
# # "e04k_t1",  # Chás, cafés, bebidas à base de ervas (chás naturais ou Lipton, Leão, Ice Tea, Do Bem, chás de latinha- branco, verde)" 1	Sim 0	Não 99	Não sei
# # "e04k_1_t1",  # Se SIM, essas bebidas tinham açúcar ou adoçante? 1	Sim 0	Não 99	Não sei
# # "e04l_t1",  # Caldos ou Sopas    1	Sim 0	Não 99	Não sei
# # "e04m_t1",  # Algum outro tipo de líquido?   1	Sim 0	Não 99	Não sei
# # "e04m_2_t1",  # Se SIM, a esses líquidos foi adicionado açúcar?  1	Sim 0	Não 99	Não sei
# # "e05b_t1",  # Arroz, pão, macarrão, mingau, cuscuz, milho/fubá, aveia, quinoa    1	Sim 0	Não 99	Não sei
# # "e05c_t1",  # Abóbora, cenoura, batatas doces, pimentões vermelhos e amarelos    1	Sim 0	Não 99	Não sei
# # "e05d_t1",  # Batata branca, mandioca, inhame, banana da terra   1	Sim 0	Não 99	Não sei
# # "e05e_t1",  # Brócolis, espinafre, chicória, rúcula, escarola, couve, mostarda   1	Sim 0	Não 99	Não sei
# # "e05f_t1",  # Outros vegetais (couve-flor, alface, tomate, beterraba, chuchu, abobrinha, berinjela, pepino)  1	Sim 0	Não 99	Não sei
# # "e05g_t1",  # Frutas (laranja, banana, maçã, manga, papaia, mamão, pêssego, melão, caqui, ameixa, uva, kiwi, mexerica, figo, pera, abacaxi, melancia)"   1	Sim 0	Não 99	Não sei
# # "e05h_t1",  # Vísceras (fígado, rins)    1	Sim 0	Não 99	Não sei
# # "e05i_t1",  # Hambúrguer, salsichas, linguiças, presunto, mortadela, bacon, salames  1	Sim 0	Não 99	Não sei
# # "e05j_t1",  # Carnes, todos os tipos: vaca, porco, frango    1	Sim 0	Não 99	Não sei
# # "e05k_t1",  # Ovos   1	Sim 0	Não 99	Não sei
# # "e05l_t1",  # Peixes 1	Sim 0	Não 99	Não sei
# # "e05m_t1",  # Feijão, ervilha, lentilha  1	Sim 0	Não 99	Não sei
# # "e05n_t1",  # Queijos    1	Sim 0	Não 99	Não sei
# # "e05o_t1",  # Chocolates, doces, tortas, bolos, biscoitos (salgado, doce ou recheado), sorvetes, gelatinas, balas, pirulitos, chicletes - industrializados 1	Sim 0	Não 99	Não sei
# # "e05p_t1",  # Doces, biscoitos (doces ou salgados), tortas, bolos, sorvetes feitos em casa 1	Sim 0	Não 99	Não sei
# # "e05q_t1",  # Batatas fritas, massas fritas (inclusive salgadinhos), macarrão instantâneo, folhados, salgadinhos de pacote"  1	Sim 0	Não 99	Não sei
# # "e05r_t1",  # Outros tipos de alimentos sólidos, semi sólidos ou pastosos.   1	Sim 0	Não 99	Não sei
# # "e06_t1",  # Quantas refeições o seu filho (a) fez da manhã de ontem até a manhã de hoje?
# # "f01_t1",  # Seu bebê teve suspeita ou confirmação de problema visual?
# # "f02_t1",  # Seu bebê teve suspeita ou confirmação de problema auditivo?
# # "f03_t1",  # Seu bebê teve suspeita ou confirmação de algum problema de saúde detectado no teste do pezinho?
# # "f04a_t1",  # Seu bebê já teve alguma infecção, como infecção de ouvido, garganta, pulmão, etc?"
# # "f05_t1",  # Seu bebê já recebeu algum diagnóstico médico (exceto infecções), como refluxo gastroesofágico, alergia alimentar, problema ósseo ou muscular, problema neurológico, cardíaco ou qualquer outro?"
# # "f05_qtde_t1",  # Quantos diagnósticos?
# # "f06_t1",  # Seu bebê já teve qualquer tipo de acidente (como queda, queimadura, corte, batida na cabeça) que precisou ser levado para emergência ou pronto atendimento?
# # "f06a_t1",  # Se sim, quantas vezes?
# # "f06b_t1",  # Se sim, qual medicação foi prescrita?
# # "f08_t1",  # Quem tem sido o principal cuidador do bebe nesses 3 primeiros meses?
# # "f09___1_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Mãe)"
# # "f09___2_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Pai)"
# # "f09___3_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Avós maternas)"
# # "f09___4_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Avós paternos)"
# # "f09___5_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Irmão)"
# # "f09___6_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Tio)"
# # "f09___7_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Outro familiar)"
# # "f09___8_t1",  # Quem são as pessoas que ajudam você nos cuidados do seu bebê nesses 3 primeiros meses? (choice=Amigo)"
# # "f09___9_t1",  # Quem são as pessoas que ajudam você nos cuidados "f09___10_t1",# : Quem são as pessoas que ajudam você nos cuidadosdo seu bebê nesses 3 primeiros meses? (choice=Empregado doméstico)"
# # "f10___2_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Pai)
# # "f10___3_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Avós maternas)
# # "f10___4_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Avós paternos)
# # "f10___5_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Irmão)
# # "f10___6_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Tio)
# # "f10___7_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Outro familiar)
# # "f10___8_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Amigo)
# # "f10___9_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Babá ou enfermeira)
# # "f10___10_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Empregado doméstico)
# # "f10___99_t1",  # : Quem são as pessoas que ficam responsáveis pelo seu bebê para você descansar ou sair de casa nesses 3 primeiros meses? (choice=Não sei)
# # "f11_t1",  # : Seu bebê frequenta creche ou fica na casa de outra pessoa (como vizinha) que recebe para isso?
# # "f12_t1",  # Onde mora o pai do bebê atualmente?
# # "f13_t1",  # : Na maior parte da gestação, você sentiu que podia contar com o pai do seu bebê como suporte emocional?
# # "f14_t1",  # Desde o nascimento do bebê, na maior parte do tempo, você sente que pode contar com o pai do seu bebê como suporte emocional?
# # "f15_t1",  # Na maior parte da gestação, você sentiu que podia contar com o pai do seu bebê como suporte financeiro ou para as necessidades do dia a dia (como levar ao médico, preparar o quarto)?
# # "f16_t1",  # : Desde o nascimento do bebê, na maior parte do tempo, você sente que pode contar com o pai do seu bebê como suporte financeiro ou para as necessidades do dia a dia (como levar ao médico, alimentar)?
# # "f17_t1",  # Caso o pai não more na mesma casa. Com que frequência ele vê (visita) o bebê?
# # "f18_t1",  # Tem alguém na casa que desempenhe papel de ''pai'' para seu bebê? (branching logic na pergunta 57 ''onde mora o pai do bebe hoje?'')
# # "bmi_t1",  # BMI based on length (f20) and weight (f19)
# # "bmz_t1",  # BMI z-score based on age (idade_crianca_dias) and WHO Child Growth Standards
# # "haz_t1",  # Height z-score based on age (idade_crianca_dias) and WHO Child Growth Standards
# # "waz_t1",#  Weight z-score based on age (idade_crianca_dias) and WHO Child Growth Standards
# # "bmzCat_t1",#  BMI z-score classification
# # "whz_t1",  # Weight-for-length z-score based on age (idade_crianca_dias) and WHO Child Growth Standards
# # "hcz_t1",#  Head circumference z-score based on age (idade_crianca_dias) and WHO Child Growth Standards
# # "bisq_2_t1",  # Em que posição seu(sua) filho(a) dorme na maior parte das vezes?
# # "bisq_3_t1",#  Quanto tempo seu(sua) filho(a) passa dormindo durante a NOITE (entre 7 da noite e 7 da manhã)?
# # "bisq_5_t1",#  Média de vezes que seu (sua) filho(a) acorda por noite
# # "bisq_6_t1",#  Durante a noite (entre 10 da noite e 6 da manhã) quanto tempo seu filho permanece acordado(a)?
# # "bisq_7_t1",#  Quanto tempo você leva para fazer seu(sua) filho(a) adormece à noite?
# # "bisq_8_t1",#  Como o seu bebê adormece?    1	Sendo alimentado    2	Sendo embalado  3	No colo 4	Sozinho na sua cama 5	Na cama perto dos pais
# # "bisq_10_t1",#  Você considera o sono do seu(sua) filho(a) um problema?
"""
