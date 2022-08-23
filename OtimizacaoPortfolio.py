# Aula 1- Alocação de ativos de maneira randômica
# -*- coding: utf-8 -*-
# precisei instalar sudo apt-get install python3-tk e openpyxl
# PROBLEMAS:
#           1-ALGORÝTMO GENÉTICO

#import scipy.optimize as solver
import pandas as pd
import matplotlib.pyplot as fig
import numpy as np
import pandas_datareader.data as web
import datetime as dt
import plotly.express as px
import six
import sys
sys.modules['sklearn.externals.six'] = six
import mlrose

# Coletando dados e montando EXEL
inicio = dt.datetime(2015, 1, 1)
fim = dt.datetime(2022, 7, 7)
list = ['BBAS3.SA', 'BBDC4.SA', 'ITUB4.SA', 'SANB11.SA', 'ABCB4.SA', 'BPAC11.SA', 'BPAN4.SA']
df = web.DataReader(list, 'yahoo', inicio, fim)['Adj Close']
writer = pd.ExcelWriter('dados.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()


dataset = pd.read_excel('aula1.xlsx', sheet_name='Sheet1')
len(dataset.columns) - 1
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(dataset.loc[len(dataset) - 1]['BBDC4.SA'])


def alocacao_ativos(dataset, dinheiro_total, seed=100, melhores_pesos=[]):
    dataset = dataset.copy()

    if seed != 0:
        np.random.seed(seed)

    if len(melhores_pesos) > 0:
        pesos = melhores_pesos
    else:
        pesos = np.random.random(len(dataset.columns) - 1)
        pesos = pesos / pesos.sum()

    colunas = dataset.columns[1:]

    for d in colunas:
        dataset[d] = (dataset[d] / dataset[d][0])

    for t, acao in enumerate(dataset.columns[1:]):
        dataset[acao] = dataset[acao] * pesos[t] * dinheiro_total

    dataset['soma valor'] = dataset.sum(axis=1, numeric_only=True)

    datas = dataset['Date']

    dataset.drop(labels=['Date'], axis=1, inplace=True)
    dataset['taxa retorno'] = 0.0

    for i in range(1, len(dataset)):
        dataset['taxa retorno'][i] = ((dataset['soma valor'][i] / dataset['soma valor'][i - 1]) - 1) * 100

    acoes_pesos = pd.DataFrame(data={'Ações': colunas, 'Pesos': pesos * 100})

    return dataset, datas, acoes_pesos, dataset.loc[len(dataset) - 1]['soma valor']
# DEFINIR VALORES ABAIXO\/\/\/\/\/\/\/


dataset, datas, acoes_pesos, soma_valor = alocacao_ativos(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, 10)


print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(dataset)  # lista de ativos e seus precos
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(datas)  # numero de dias analisados
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(acoes_pesos)  # peso das acoes, quase que aleatorio
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Valor total final: R$%0.2f' % soma_valor)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# +++++++++++++++++++++++++++++++++++++++++++++++++++ ( IMPRIMINDO AS FIGURAS ) ++++++++++++++++++++++++++++++++++++++ #
figura = px.line(x=datas, y=dataset['taxa retorno'], title='Retorno diário do portfólio')
figura.show()

figura = px.line(title='Evolução do patrimônio')
for i in dataset.drop(columns=['soma valor', 'taxa retorno']).columns:
    figura.add_scatter(x=datas, y=dataset[i], name=i)
figura.show()

figura = px.line(x=datas, y=dataset['soma valor'], title='Evolução do patrimônio')
figura.show()

# +++++++++++++++++++++++++++++++++++++++++++++++ ( ANÝLISE ESTATÝSTICA PERÝODO ) ++++++++++++++++++++++++++++++++++++ #
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Retorno Acumulado no Período: ',  dataset.loc[len(dataset) - 1]['soma valor'] / dataset.loc[0]['soma valor'] - 1)
print('Desvio Padrão: ', dataset['taxa retorno'].std())
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# +++++++++++++++++++++++++++++++++++++++++++++++++++ ( SHARPER RATIO) ++++++++++++++++++++++++++++++++++++++++ #
(dataset['taxa retorno'].mean() / dataset['taxa retorno'].std()) * np.sqrt(246)
dinheiro_total = 5000
print('Diferença entre valor investido-valor atual: R$%0.2f' % (soma_valor - dinheiro_total))
# ++++++++++++++++++++++++++++++++++++++++++++++ ( DEFININDO VALORES TAXA SELIC ) +++++++++++++++++++++++++++++++++++ #
taxa_selic_2015 = 12.75
taxa_selic_2016 = 14.25
taxa_selic_2017 = 12.25
taxa_selic_2018 = 6.50
taxa_selic_2019 = 5.0
taxa_selic_2020 = 2.0
taxa_selic_2021 = 9.15
# ++++++++++++++++++++++++++++ ( CALCULANDO INVESTIMENTO RENDA FIXA SELIC--> RISK FREE ) ++++++++++++++++++++++++++++ #
valor_2015 = dinheiro_total + (dinheiro_total * taxa_selic_2015 / 100)
valor_2016 = valor_2015 + (valor_2015 * taxa_selic_2016 / 100)
valor_2017 = valor_2016 + (valor_2016 * taxa_selic_2017 / 100)
valor_2018 = valor_2017 + (valor_2017 * taxa_selic_2018 / 100)
valor_2019 = valor_2018 + (valor_2018 * taxa_selic_2019 / 100)
valor_2020 = valor_2019 + (valor_2019 * taxa_selic_2020 / 100)
valor_2021 = valor_2020 + (valor_2020 * taxa_selic_2021 / 100)
rendimento = valor_2020 - dinheiro_total
ir = rendimento * 15 / 100
totalLiquido = valor_2021
lucroLiquido = valor_2021 - ir - dinheiro_total
taxa_selic_historico = np.array([12.75, 14.25, 12.25, 6.5, 5.0, 2.0, 9.15])
taxa_selic_historico.mean() / 100
# ++++++++++++++++++++++++++++++++++++++++++ ( CRESCIMENTO PATRIMONIAL RENDA FIXA ) ++++++++++++++++++++++++++++++++++ #
print('Valor em 2015: R$%0.2f' % valor_2015)
print('Valor em 2016: R$%0.2f' % valor_2016)
print('Valor em 2017: R$%0.2f' % valor_2017)
print('Valor em 2018: R$%0.2f' % valor_2018)
print('Valor em 2019: R$%0.2f' % valor_2019)
print('Valor em 2020: R$%0.2f' % valor_2020)
print('Valor em 2021: R$%0.2f' % valor_2021)
print('Lucro Bruto investimento: R$%0.2f' % rendimento)
print('Total de imposto pago R$%0.2f' % ir)
print('Total Liquido investimento: R$%0.2f' % totalLiquido)
print('Lucro Liquido investimento: R$%0.2f' % lucroLiquido)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Taxa Selic média ultimos 7 anos %0.2f' % taxa_selic_historico.mean(), '%')
print('Sharp Ratio sem considerar Renda Fixa: ',  (dataset['taxa retorno'].mean() / dataset['taxa retorno'].std()) * np.sqrt(246))
print('Sharp Ratio considerando Renda Fixa: ', (dataset['taxa retorno'].mean() - taxa_selic_historico.mean() / 100) / dataset['taxa retorno'].std() * np.sqrt(246))

# 1<SR<2= BOM. 2<SR<3 = MUITO BOM. 3<SR<4 = excelente #

# +++++++++++++++++++++++++++++++++++++++++ ( OTIMIZAÇÃO DE PORTFÓLIO RANDÔMICA ) ++++++++++++++++++++++++++++++++++++ #


def alocacao_portfolio(dataset, dinheiro_total, sem_risco, repeticoes):

    dataset = dataset.copy()
    dataset_original = dataset.copy()

    lista_retorno_esperado = []
    lista_volatilidade_esperada = []
    lista_sharpe_ratio = []

    melhor_sharpe_ratio = 1 - sys.maxsize
    melhores_pesos = np.empty
    melhor_volatilidade = 0
    melhor_retorno = 0

    for _ in range(repeticoes):
        pesos = np.random.random(len(dataset.columns) - 1)
        pesos = pesos / pesos.sum()

        for i in dataset.columns[1:]:
            dataset[i] = dataset[i] / dataset[i][0]

        for i, acao in enumerate(dataset.columns[1:]):
            dataset[acao] = dataset[acao] * pesos[i] * dinheiro_total

        dataset.drop(labels=['Date'], axis=1, inplace=True)

        retorno_carteira = np.log(dataset / dataset.shift(1))
        matriz_covariancia = retorno_carteira.cov()

        dataset['soma valor'] = dataset.sum(axis=1)
        dataset['taxa retorno'] = 0.0

        for i in range(1, len(dataset)):
            dataset['taxa retorno'][i] = np.log(dataset['soma valor'][i] / dataset['soma valor'][i - 1])

        retorno_esperado = np.sum(dataset['taxa retorno'].mean() * pesos) * 246
        volatilidade_esperada = np.sqrt(np.dot(pesos, np.dot(matriz_covariancia * 246, pesos)))
        sharpe_ratio = (retorno_esperado - sem_risco) / volatilidade_esperada

        if sharpe_ratio > melhor_sharpe_ratio:
            melhor_sharpe_ratio = sharpe_ratio
            melhores_pesos = pesos
            melhor_volatilidade = volatilidade_esperada
            melhor_retorno = retorno_esperado

        lista_retorno_esperado.append(retorno_esperado)
        lista_volatilidade_esperada.append(volatilidade_esperada)
        lista_sharpe_ratio.append(sharpe_ratio)

        dataset = dataset_original.copy()

    return melhor_sharpe_ratio, melhores_pesos, lista_retorno_esperado, lista_volatilidade_esperada, lista_sharpe_ratio, melhor_volatilidade, melhor_retorno
# +++++++++++++++++++++++++++++++ ( VOCÊ MUDA OS PARÂMETROS NO CHAMADO DA FUNÇÃO ABAIXO ) ++++++++++++++++++++++++++++ #


sharpe_ratio, melhores_pesos, ls_retorno, ls_volatilidade, ls_sharpe_ratio, melhor_volatilidade, melhor_retorno = alocacao_portfolio(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, taxa_selic_historico.mean()/100, 5000)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Sharp Ratio:', sharpe_ratio)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Melhores Pesos: ', melhores_pesos)
# pd.read_excel('aula1.xlsx', sheet_name='Sheet1')
_, _, acoes_pesos, soma_valor = alocacao_ativos(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, 100, melhores_pesos=melhores_pesos)
# ++++++++++++++++++++++++++++ ( VOCÊ MUDA OS PARÂMETROS NO CHAMADO DA FUNÇÃO ACIMA) +++++++++++++++++++++++++++++++ #
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Sharp Ratio: %0.2f' % sharpe_ratio)  # lista de ativos e seus precos
print('Melhores Pesos: ', melhores_pesos)  # numero de dias analisados
print('Pesos de cada Ação: ', acoes_pesos)  # peso das acoes, quase que aleatorio
print('Valor total: R$%0.2f' % soma_valor)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Melhor Retorno: ', melhor_retorno, '%')
print('Melhor Volatilidade: ', melhor_volatilidade, '%')
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
# ++++++++++++++++++++++++++++++++++++++++++++++++ ( PLOTAR GRÝFICO BONITINHO AI ) ++++++++++++++++++++++++++++++++++ #

fig.figure(figsize=(10, 8))
fig.scatter(ls_volatilidade, ls_retorno, c=ls_sharpe_ratio)
fig.colorbar(label='Sharpe ratio')
fig.xlabel('Volatilidade')
fig.ylabel('Retorno')
fig.scatter(melhor_volatilidade, melhor_retorno, c='red', s=100)
fig.show()
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ( FUNÇÃO OTIMIZAÇÃO ) +++++++++++++++++++++++++++++++++++ #

dataset_original = pd.read_excel('aula1.xlsx', sheet_name='Sheet1')
sem_risco = taxa_selic_historico.mean() / 100


def fitness_function(solucao):
    dataset = dataset_original.copy()
    pesos = solucao / solucao.sum()

    for s in dataset.columns[1:]:
        dataset[s] = (dataset[s] / dataset[s][0])

    for v, acao in enumerate(dataset.columns[1:]):
        dataset[acao] = dataset[acao] * pesos[v] * dinheiro_total

    dataset.drop(labels=['Date'], axis=1, inplace=True)
    dataset['soma valor'] = dataset.sum(axis=1)
    dataset['taxa retorno'] = 0.0

    for c in range(1, len(dataset)):
        dataset['taxa retorno'][c] = ((dataset['soma valor'][c] / dataset['soma valor'][c - 1]) - 1) * 100

    sharpe_ratio = (dataset['taxa retorno'].mean() - sem_risco) / dataset['taxa retorno'].std() * np.sqrt(246)

    return sharpe_ratio


np.random.seed(100)
pesos = np.random.random(len(dataset_original.columns) - 1)
pesos = pesos / pesos.sum()

fitness_function(pesos)


def visualiza_alocacao(solucao):
    colunas = dataset_original.columns[1:]
    for u in range(len(solucao)):
        print(colunas[u], solucao[u] * 100)


visualiza_alocacao(pesos)
fitness = mlrose.CustomFitness(fitness_function)
problema_maximizacao = mlrose.ContinuousOpt(length=6, fitness_fn=fitness, maximize=True, min_val=0, max_val=1)
problema_minimizacao = mlrose.ContinuousOpt(length=6, fitness_fn=fitness, maximize=False, min_val=0, max_val=1)

print('++++++++++++++++++++++++++++ ( MELHOR SOLUÇÃO - Hill Climb  ) +++++++++++++++++++++++++++++++++++++++++++')
melhor_solucao, melhor_custo = mlrose.hill_climb(problema_maximizacao, random_state=1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
print('Melhor Solucao: ', melhor_solucao)
print('Melhor Custo: ', melhor_custo)
visualiza_alocacao(melhor_solucao)
_, _, _, soma_valor = alocacao_ativos(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, melhores_pesos=melhor_solucao)
print('SOMA VALOR: ', soma_valor)

print('++++++++++++++++++++++++++++ ( PIOR SOLUÇÃO - Hill Climb ) +++++++++++++++++++++++++++++++++++++++++++')
pior_solucao, pior_custo = mlrose.hill_climb(problema_minimizacao, random_state=1)
pior_solucao = pior_solucao / pior_solucao.sum()
print('Pior Solucao: ', pior_solucao)
print('Pior Custo: ', pior_custo)
visualiza_alocacao(pior_solucao)
_, _, _, soma_valor = alocacao_ativos(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, melhores_pesos=pior_solucao)
print('SOMA VALOR: ', soma_valor)

print('++++++++++++++++++++++++++++ ( MELHOR CUSTO -Simulated annealing ) +++++++++++++++++++++++++++++++++++++++++++')
melhor_solucao, melhor_custo = mlrose.simulated_annealing(problema_maximizacao, random_state=1)
melhor_solucao = melhor_solucao / melhor_solucao.sum()
print('Melhor Solucao: ', melhor_solucao)
print('Melhor Custo: ', melhor_custo)
visualiza_alocacao(melhor_solucao)
_, _, _, soma_valor = alocacao_ativos(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, melhores_pesos=melhor_solucao)
print('SOMA VALOR: ', soma_valor)


print('++++++++++++++++++++++++++++++++ ( ALGORITMO GENETICO ) +++++++++++++++++++++++++++++++++++++++++++')
problema_maximizacao_ag = mlrose.ContinuousOpt(length=6, fitness_fn=fitness, maximize=True, min_val=0.1, max_val=1)
melhor_solucao_ag, melhor_custo_ag = mlrose.genetic_alg(problema_maximizacao_ag, random_state=1)
melhor_solucao_ag = melhor_solucao_ag / melhor_solucao_ag.sum()
print('Solucao AG: ', melhor_solucao_ag)
print('Custo AG: ', melhor_custo_ag)
visualiza_alocacao(melhor_solucao_ag)
_, _, _, soma_valor = alocacao_ativos(pd.read_excel('aula1.xlsx', sheet_name='Sheet1'), 5000, melhores_pesos=melhor_solucao_ag)
print('SOMA VALOR: ', soma_valor)
