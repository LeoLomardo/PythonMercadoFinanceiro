import pandas as pd
import numpy as np
import plotly.express as px
import seaborn as sns
from scipy import stats
import pandas_datareader.data as web
import datetime as dt
from sklearn.metrics import mean_absolute_error

# ++++++++++++++++++++++++++ ( COLETANDO DADOS E MONTANDO EXEL ) ++++++++++++++++++++++++++++ #
dia0 = dt.datetime(2015, 1, 1)
fim = dt.datetime(2022, 7, 7)
list = ['GGBR4.SA', 'BBDC4.SA', 'EMBR3.SA', 'ENBR3.SA', 'PETR4.SA', 'BOVA11.SA']
df = web.DataReader(list, 'yahoo', dia0, fim)['Adj Close']
writer = pd.ExcelWriter('aula3.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()
dataset = pd.read_excel('dados.xlsx', sheet_name='Sheet1')


# ++++++++++++++++++++++++++ ( SIMULACAO DE MONTE CARLO ) ++++++++++++++++++++++++++++ #

def monte_carlo_previsao(dataset, ativo, dias_a_frente, simulacoes):
    dataset = dataset.copy()
    dataset = pd.DataFrame(dataset[ativo])

    dataset_normalizado = dataset.copy()
    for i in dataset:
        dataset_normalizado[i] = dataset[i] / dataset[i][0]

    dataset_taxa_retorno = np.log(1 + dataset_normalizado.pct_change())
    dataset_taxa_retorno.fillna(0, inplace=True)

    media = dataset_taxa_retorno.mean()
    variancia = dataset_taxa_retorno.var()

    drift = media - (0.5 * variancia)
    desvio_padrao = dataset_taxa_retorno.std()
    Z = stats.norm.ppf(np.random.rand(dias_a_frente, simulacoes))
    retornos_diarios = np.exp(drift.values + desvio_padrao.values * Z)

    previsoes = np.zeros_like(retornos_diarios)
    previsoes[0] = dataset.iloc[-1]

    for dia in range(1, dias_a_frente):
        previsoes[dia] = previsoes[dia - 1] * retornos_diarios[dia]

    figura = px.line(title='Previsões do preço das ações - ' + ativo)
    for i in range(len(previsoes.T)):
        figura.add_scatter(y=previsoes.T[i], name=i)
    figura.show()

    return previsoes.T


for ativo in dataset.columns[1:]:
    monte_carlo_previsao(dataset, ativo, 30, 100)

# TENTAR IMPLEMENTAR PLOTAGEM GRÁFICO PIOR SIMULAÇÃO X MELHOR SIMULAÇÃO X VALOR REAL#
print('OLA GITHUB')
