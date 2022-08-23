import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import datetime

import plotly.io
from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima.arima import auto_arima
import pandas_datareader.data as web
import datetime as dt
from sklearn.metrics import mean_absolute_error
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly

# ESSE CÓDIGO É MUITO BOM PARA CASOS ONDE SE TEM UMA BASE DE DADOS MUITO GRANDE
# ++++++++++++++++++++++++++ ( COLETANDO DADOS E MONTANDO EXEL ) ++++++++++++++++++++++++++++ #
dia0 = dt.datetime(2015, 1, 1)
fim = dt.datetime(2022, 7, 7)
list = ['EMBR3.SA', 'BBDC4.SA', 'EMBR3.SA', 'ENBR3.SA', 'PETR4.SA', 'BOVA11.SA']
df = web.DataReader(list, 'yahoo', dia0, fim)['Adj Close']
writer = pd.ExcelWriter('aula4.xlsx', engine='xlsxwriter')
df.to_excel(writer, sheet_name='Sheet1')
writer.save()

# ++++++++++++++++++++++++++ ( PLOTANDO GRÁFICO) ++++++++++++++++++++++++++++++++++++++++++ #
dateparse = lambda dates: datetime.datetime.strptime(dates, '%Y-%m-%d %H:%M:%S')
dataset = pd.read_excel('aula4.xlsx', parse_dates=['Date'], index_col='Date', date_parser=dateparse, usecols=['Date', 'BOVA11.SA'])
time_series = dataset['BOVA11.SA']
plt.plot(time_series)
plt.show()
figura = px.line(title='Histórico do preço das ações')
figura.add_scatter(x=time_series.index, y=time_series)
figura.show()
# ++++++++++++++++++++++++++ ( DECOMPONDO A SÉRIE TEMPORAL ) +++++++++++++++++++++++++++++++++++++++++ #
type(time_series)
len(time_series) / 2
decomposicao = seasonal_decompose(time_series, period=len(time_series) // 2)
tendencia = decomposicao.trend
sazonal = decomposicao.seasonal
aleatorio = decomposicao.resid
plt.plot(tendencia)
plt.show()
plt.plot(sazonal)
plt.show()
plt.plot(aleatorio)
plt.show()
# ++++++++++++++++++++++++++ ( PREVISÃO COM ARIMA - ESTIMADOR FRACO ) +++++++++++++++++++++++++++++++++++++++++++++++++ #
modelo = auto_arima(time_series, suppress_warnings=True, error_action='ignore')
# Parâmetros P, Q e D
print(modelo.order)
previsoes = modelo.predict(n_periods=15)
print(previsoes)
valor_base = len(time_series)-365

treinamento = time_series[:valor_base]
teste = time_series[valor_base:]
modelo2 = auto_arima(treinamento, suppress_warnings=True, error_action='ignore')
previsoes = pd.DataFrame(modelo2.predict(n_periods=365), index=teste.index)
previsoes.columns = ['previsoes']
print(previsoes)

plt.figure(figsize=(8, 5))
plt.plot(treinamento, label='Treinamento')
plt.show()
plt.plot(teste, label='Teste')
plt.show()
plt.plot(previsoes, label='Previsões')
plt.legend()
plt.show()
print(teste.index)
# ++++++++++++++++++++++++++ ( AVALIANDO PREVISÃO COM ARIMA ) +++++++++++++++++++++++++++++++++++++++++++++++++ #
print('Erro precificação: +-R$', mean_absolute_error(teste, previsoes))

# ++++++++++++++++++++++++++ ( PREVISÃO COM FACEBOOK PROFETH -  ESTIMADOR MÉDIO ) +++++++++++++++++++++++++++++++++++++++++++++++++ #
dataset_fb = pd.read_excel('aula4.xlsx', usecols=['Date', 'BOVA11.SA'])
dataset_fb = dataset_fb[['Date', 'BOVA11.SA']].rename(columns={'Date': 'ds', 'BOVA11.SA': 'y'})

modelo2 = Prophet()
modelo2.fit(dataset_fb)  # Aqui passamos os dados para o programa "treinar"

futuro = modelo2.make_future_dataframe(periods=0) # Aqui ele monta um novo dataframe com as previsões, no caso, ele gera previsões para as datas iniciais + 15 futuras
previsoes2 = modelo2.predict(futuro)
fig1 = modelo2.plot(previsoes2)
fig1.show()
fig2 = modelo2.plot_components(previsoes2)
fig2.show()
fig3 = plot_plotly(modelo2, previsoes2)
fig3.show()
fig5 = plot_components_plotly(modelo2, previsoes2)
fig5.show()
len(dataset_fb)  # tamanho da base de dados de registros
len(previsoes2)  # tamanho da base de dados de registro + o número de previsões desejadas

tam_previstos = len(previsoes2) - len(dataset_fb)

print(previsoes2.tail(tam_previstos))
fig4 = modelo2.plot(previsoes2, xlabel='Data', ylabel='Preço')
fig4.show()

# ++++++++++++++++++++++++++ ( AVALIANDO PREVISÃO FBPROPHET ) +++++++++++++++++++++++++++++++++++++++++++++++++ #
pred = modelo2.make_future_dataframe(periods=0)
previsoes2 = modelo2.predict(pred)
print(previsoes2.shape)
print(previsoes2)
previsoes2 = previsoes2['yhat'].tail(365)
print('Erro precificação: +-R$', mean_absolute_error(teste, previsoes2))
