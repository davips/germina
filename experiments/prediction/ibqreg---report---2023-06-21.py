from time import sleep

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from xgboost import XGBClassifier as XGBc
from catboost import CatBoostClassifier as CATc
from lightgbm import LGBMClassifier as LGBMc
from sklearn.linear_model import LogisticRegression as LRc
from sklearn.ensemble import ExtraTreesClassifier as ETc
from sklearn.linear_model import SGDClassifier as SGDc
import pandas as pd
from scipy.stats import gmean
from sklearn import clone
from sklearn.inspection import permutation_importance

from germina.ordinalclassifier import OrdinalClassifier
from numpy import log, ndarray
from pandas import DataFrame
from scipy.spatial.distance import cdist
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from germina.dataset import join
from hdict import _, apply, cache, hdict, field
from hdict.dataset.pandas_handling import file2df
from shelchemy import sopen, memory
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

from germina.config import local_cache_uri, remote_cache_uri
from germina.nan import remove_worst_nan_rows, backup_cols, hasNaN, remove_worst_nan_cols, remove_cols, bina, loga, remove_nan_rows_cols, remove_nan_cols_rows
from hdict import _, apply, cache, hdict
from shelchemy import sopen

path = "data/"
areas = ["microbiome_alpha", "microbiome_pathways", "microbiome_species", "eeg1", "eeg2"]
metavars = [
               "id_estudo",
               "risco_class",
               "epds_tot_t1",  # EPDS Total Score
               "chaos_tot_t1",
               "renda_familiar_total_t0",
               "delivery_mode",
               "EBF_3m",
               "infant_ethinicity",

               "idade_crianca_meses_t1",
               "idade_crianca_meses_t2",
           ] + [
               # "c12f_t1",  # Depressão durante a gestação ou no pós-parto
               # "educationLevelAhmedNum_t1",
               # "elegib14_t0",  # sexo
               # "elegib2_t0",  # idade mãe
               # "a08_t1",  # Etnia da criança
               # "renda_t2",
               # "renda_total_t2",
               # "maternal_ethinicity",
               # "antibiotic",
               # "risco_total_t0",
               # "risco_class_cod",
               #
           ] + [
               # "b04_t1",  # Etnia da mãe
               # "b13_t1",  # Etnia do pai
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
           ]
targets = ["ibq_reg_t1", "ibq_reg_t2", ]  # "bayley_17_t1",    # "bayley_17_t2"

with sopen(local_cache_uri) as local, sopen(remote_cache_uri) as remote:
    d = hdict(random_state=0, return_name=False, index="id_estudo", keep=["id_estudo"] + targets)

    # metadata ##################################################################################################################
    d >>= apply(file2df, path + "metadata___2023-06-18.csv").metadata
    d >>= apply(DataFrame.__getitem__, _.metadata, metavars + targets).df
    d >>= apply(remove_nan_rows_cols, rows_at_a_time=4).df
    print("Format problematic attributes.")
    d >>= apply(bina, attribute="antibiotic", positive_category="yes").df
    d >>= apply(bina, attribute="EBF_3m", positive_category="EBF").df
    d >>= apply(loga, attribute="renda_familiar_total_t0").df

    # microbiome #################################################################################################################
    d >>= apply(file2df, path + "data_microbiome___2023-06-18___alpha_diversity_n525.csv").microbiome_alpha
    d >>= apply(join, other=_.microbiome_alpha).df
    # d >>= apply(file2df, path + "data_microbiome___2023-06-20___vias_metabolicas_3_meses_n525.csv").microbiome_pathways
    # d >>= apply(join, other=_.microbiome_pathways).df
    # d >>= apply(file2df, path + "data_microbiome___2023-06-18___especies_3_meses_n525.csv").microbiome_species
    # d >>= apply(join, other=_.microbiome_species).df
    # d >>= apply(remove_nan_rows_cols).df

    # eeg ########################################################################################################################
    d >>= apply(file2df, path + "data_eeg___2023-06-20___T1_RS_average_dwPLI_withEEGCovariates.csv").eeg1
    d >>= apply(join, other=_.eeg1).df
    d >>= apply(file2df, path + "data_eeg___2023-06-20___T2_RS_average_dwPLI_withEEGCovariates.csv").eeg2
    d >>= apply(join, other=_.eeg2).df
    d >>= apply(remove_nan_rows_cols, rows_at_a_time=2).df  # EEG minattrs e maxattrs:   rows_at_a_time=2→(452, 15)    rows_at_a_time=3→(293, 28)

    # d >>= cache(remote) >> cache(local)
    d >>= cache(local) >> cache(remote)

    # Visualize ####################################################################################################################
    # d.df.to_csv(f"/tmp/{'-'.join(areas + ['metadata'])}.csv")
    # d.df: DataFrame
    # for target in targets:
    #     d.df[target].hist(bins=3)
    # plt.show()
    # TODO criar trigger para fazer log quando chama um field (seja cacheado ou não)
    #   _ pode ser arg ou kwarg padrão para apontar para o hdict: lambda x,_,z: _.r+x
    #
    d >>= apply(lambda df: print("df:", df.shape)).log
    d >>= apply(lambda df, log: print("df:", list(df.columns))).log
    sleep(2)

    # Train #######################################################################################################################
    d >>= apply(KFold, n_splits=8, random_state=0, shuffle=True).cv
    for target in targets:
        print("=======================================================")
        print(target)
        print("=======================================================")

        # Prepare dataset.
        cuts = [5, 6]
        d >>= apply(getattr, _.df, target).t
        d >>= apply(lambda x, cuts: np.digitize(x, cuts), _.t, cuts).t
        d >>= apply(lambda df, t: df[t != 1], _.df, _.t).df
        d >>= apply(remove_cols, _.df, targets, []).X
        d >>= apply(getattr, _.df, target).y
        d >>= apply(lambda x, cuts: np.digitize(x, cuts), _.y, cuts).y
        d >>= apply(lambda y: y // 2, _.y).y
        d >>= apply(lambda X, y, log: (print("X:", X.shape), print("y:", y.shape))).log
        sleep(2)

        clas_names = []
        d["n_jobs"] = -1
        d["n_estimators"] = 10000
        clas = {
            DummyClassifier: {},
            RFc: {},
            XGBc: {},
            # CATc: {"subsample": 0.1},
            LGBMc: {},
            ETc: {},
            SGDc: {},
        }
        for cla, kwargs in clas.items():
            clas_names.append(cla.__name__)
            print(clas_names[-1])
            d >>= apply(cla, **kwargs)(clas_names[-1])

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
        for m in scos:
            print(m)
            print("----------------------------------------------------")
            for classifier_field in clas_names:
                field_name = f"{m}_{classifier_field}"
                field_name_cvp = f"{field_name}_cvp"
                d >>= apply(cross_val_score, field(classifier_field), _.X, _.y, cv=_.cv, scoring=m)(field_name)
                d >>= apply(cross_val_predict, field(classifier_field), _.X, _.y, cv=_.cv)(field_name_cvp)
                d >>= cache(local) >> cache(remote)

                print(f"{classifier_field:24} {mean(d[field_name]):.6f} {std(d[field_name]):.6f} \n{confusion_matrix(d.y, d[field_name_cvp])}")
            print("----------------------------------------------------")

        for classifier_field in clas_names:
            model = f"{target}_{classifier_field}_model"
            d >>= apply(lambda c, *args, **kwargs: clone(c).fit(*args, **kwargs), field(classifier_field), _.X, _.y)(model)
            importances_field_name = f"{target}_{classifier_field}_importances"
            d >>= apply(permutation_importance, field(model), _.X, _.y, n_repeats=20, scoring=scos, n_jobs=-1)(importances_field_name)
            d >>= cache(local) >> cache(remote)
            fst = True
            for metric in d[importances_field_name]:
                r = d[importances_field_name][metric]
                for i in r.importances_mean.argsort()[::-1]:
                    if r.importances_mean[i] - r.importances_std[i] > 0:
                        if fst:
                            print("Importances +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
                            fst = False
                        print(f"  {metric}   {d.X.columns[i][-25:]:<8} {r.importances_mean[i]:.6f} +/- {r.importances_std[i]:.6f}")
            if not fst:
                print("+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print()
    d.log
