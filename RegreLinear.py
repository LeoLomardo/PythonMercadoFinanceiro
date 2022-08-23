import pandas as pd
import numpy as np
import plotly.express as px
import pandas_datareader.data as web
import datetime as dt


# ++++++++++++++++++++++++++ ( COLETANDO DADOS E MONTANDO EXEL ) ++++++++++++++++++++++++++++ #
dia0 = dt.datetime(2015, 1, 1)
fim = dt.datetime(2022, 7, 7)
list = ['GGBR4.SA', 'BBDC4.SA', 'EMBR3.SA', 'ENBR3.SA', 'PETR4.SA', 'BOVA11.SA']
df = web.DataReader(list, 'yahoo', dia0, fim)['Adj Close']
writer = pd.ExcelWriter('aula2.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

# ++++++++++++++++++++++++++ ( CRIANDO TABELAS COM DADOS ORGANIZADOS DA MANEIRA DESEJADA ) +++++++++++++++++++++++++++ #
dataset = pd.read_excel('aula2.xlsx', sheet_name='Sheet1')

dataset.drop(labels=['Date'], axis=1, inplace=True)
dataset_normalizado = dataset.copy()
for i in dataset.columns:
    dataset_normalizado[i] = dataset[i] / dataset[i][0]
print('Dados Normalizados:', dataset_normalizado)

dataset_taxa_retorno = (dataset_normalizado / dataset_normalizado.shift(1)) - 1
print('Taxa de Retorno: ', dataset_taxa_retorno)

dataset_taxa_retorno.fillna(0, inplace=True)
print('Taxas de Retorno:', dataset_taxa_retorno.head())

print('Taxa de Retorno anual:')
print(dataset_taxa_retorno.mean() * 246)

# ++++++++++++++++++++++++++ ( BETA com REGRESSÃO LINEAR ) ++++++++++++++++++++++++++++ #
figura = px.scatter(dataset_taxa_retorno, x='BOVA11.SA', y='BBDC4.SA', title='BOVA x MGLU')
figura.show()

beta, alpha = np.polyfit(x=dataset_taxa_retorno['BOVA11.SA'], y=dataset_taxa_retorno['BBDC4.SA'], deg=1)

print('beta:', beta, 'alpha:', alpha, 'alpha (%):', alpha * 100)

figura = px.scatter(dataset_taxa_retorno, x='BOVA11.SA', y='BBDC4.SA', title='BOVA11.SA x BBDC4.SA')
figura.add_scatter(x=dataset_taxa_retorno['BOVA11.SA'], y=beta * dataset_taxa_retorno['BOVA11.SA'] + alpha)
figura.show()

# ++++++++++++++++++++++++++ ( BETA com COV ) ++++++++++++++++++++++++++++ #
# ELE RETIRA OS ATIVOS QUE NAO QUER ANALISAR
matriz_covariancia = dataset_taxa_retorno.drop(columns=['GGBR4.SA', 'EMBR3.SA', 'ENBR3.SA', 'PETR4.SA']).cov() * 246
print(matriz_covariancia)

cov_bbdc_bova = matriz_covariancia.iloc[1, 0]
print('Covariância BOVA x BBDC:', cov_bbdc_bova)

variancia_bova = dataset_taxa_retorno['BOVA11.SA'].var() * 246
print('Variância BOVA:', variancia_bova)

beta_bbdc = cov_bbdc_bova / variancia_bova
print('BETA BBDC: ', beta_bbdc)
# ++++++++++++++++++++++++++ ( CAPM para 1 único ativo ) ++++++++++++++++++++++++++++ #

rm = dataset_taxa_retorno['BOVA11.SA'].mean() * 246
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(rm)

taxa_selic_historico = np.array([12.75, 14.25, 12.25, 6.5, 5.0, 2.0, 9.15])
rf = taxa_selic_historico.mean() / 100
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print(rf)

capm_bbdc = rf + (beta * (rm - rf))
print('CAPM do BBDC4: ', capm_bbdc)

# ++++++++++++++++++++++++++ ( BETA para todos os ativos  ) ++++++++++++++++++++++++++++ #

betas = []
alphas = []

print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
for ativo in dataset_taxa_retorno.columns[0:-1]:

    beta, alpha = np.polyfit(dataset_taxa_retorno['BOVA11.SA'], dataset_taxa_retorno[ativo], 1)
    betas.append(beta)
    alphas.append(alpha)


def visualiza_betas_alphas(betas, alphas):
    for i, ativo in enumerate(dataset_taxa_retorno.columns[0:-1]):
        print(ativo, 'beta:', betas[i], 'alpha:', alphas[i] * 100)


visualiza_betas_alphas(betas, alphas)
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('Alpha médio: ', np.array(alphas).mean() * 100)

# ++++++++++++++++++++++++++ ( CAPM para Portfólio ) ++++++++++++++++++++++++++++ #
capm_empresas = []
for i, ativo in enumerate(dataset_taxa_retorno.columns[0:-1]):

    capm_empresas.append(rf + (betas[i] * (rm - rf)))


def visualiza_capm(capm):
    for c, ativo in enumerate(dataset_taxa_retorno.columns[0:-1]):
        print(ativo, 'CAPM:', capm[c] * 100)


visualiza_capm(capm_empresas)
# DEFINIR PESOS PORTFOLIO ABAIXO
pesos = np.array([0.2, 0.2, 0.2, 0.2, 0.2])

capm_portfolio = np.sum(capm_empresas * pesos) * 100
print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('CAPM do porfólio: ', capm_portfolio)
