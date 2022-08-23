# -*- coding: utf-8 -*-
import pandas as pd
import matplotlib.pyplot as fig
import numpy as np
#precisei instalar sudo apt-get install python3-tk
import scipy.optimize as solver
import pandas_datareader.data as web
import datetime as dt
#++++++++++++++++++++++++++++++++++++++++++++ ( MONTA EXEL COM AÇÕES ) ++++++++++++++++++++++++++++++++++++++++++++++++#

#DELIMITANDO ESPACO DE TEMPO
inicio = dt.datetime(2015, 1, 2)
fim = dt.datetime(2020, 11, 3)

#VETOR COM NOME DOS ATIVOS
list = ['GOLL4.SA', 'CVCB3.SA', 'WEGE3.SA', 'MGLU3.SA', 'TOTS3.SA', 'BOVA11.SA']

len(list)
#MONTANDO MATRIZ COM VALORES, CASO QUERIA MAIS INFO, DELETAR "['Adj Close']", CASO QUEIRA SO PRECO DE FECHAMENTO, MANTER
df = web.DataReader(list, 'yahoo', inicio, fim)['Adj Close']

#ESCREVE NO ARQUIVO EXEL DESTINO
writer = pd.ExcelWriter('Teste3.xlsx', engine='xlsxwriter')

#ESCOLHE TABELA QUE IMPRIME OS DADOS
df.to_excel(writer, sheet_name='Sheet1')

#SALVA
writer.save()


#+++++++++++++++++++++++++++++++++++++++++++++ ( OTIMIZAÇÃO DA CARTEIRA ) +++++++++++++++++++++++++++++++++++++++++++++#

df = pd.read_excel('Teste3.xlsx', sheet_name='Sheet1')

fig.style.use('ggplot')

fig.figure()

ax1 = fig.subplot(111)
ax1.plot(df['Date'], df[list], '-k', alpha=0.8)
ax1.legend(list)

for i in range(len(list)):
    ax1.text(x=df.Date[-1:], y=df[list[i]][-1:], s=list[i], fontsize=12, color='k', weight='bold')

n = len(df)

prec = df.drop(['Date'], axis=1)

ri = prec/prec.shift(1)-1
mi = ri.mean()*len(df) #esses valores precisam estar corretos, referentes ao numero de linhas da tabela exel
sigma = ri.cov()*len(df)
print(' ')
print(' ')
print('++++++++++++++++++++++Matriz de Covariância da carteira++++++++++++++++++++++')
print(sigma)

vet_R = []
vet_Vol = []

for i in range(2000): #com minha carteira inicial, colocar 2000 ou 30000 passos nao fez diferenca significativo, testar para valores maiores
    # aqui voce define o peso da alocao de cada ativo da carteira
    w = np.random.random(len(list))
    w = w/np.sum(w)
    # outro exemplo de alocar
    #w =np.array([0.19524, 0.36, 0.028913, 0.1591, 0.256747])  #LEMBRANDO QUE O TOTAL PRECISA SER 1

    retorno = np.sum(w*mi)
    risco = np.sqrt(np.dot(w.T, np.dot(sigma, w)))

    vet_R.append(retorno)
    vet_Vol.append(risco)
fig.figure()

print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')
print('Retorno esperado da carteira = ', str(round(100 * retorno, 2)) + '%')
print('Risco da carteira =', str(round(100 * risco, 2)) + '%')

ax2 = fig.subplot(111)
ax2.plot(vet_Vol, vet_R, 'ok', alpha=0.2)
ax2.grid()
ax2.set_xlabel('VOLATILIDADE ESPERADA', fontsize=14)
ax2.set_ylabel("RETORNO ESPERADO", fontsize=14)
fig.title('Retorno Esperado x Volatilidade esperada', fontsize=10)
fig.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++( FRONTEIRA EFICIENTE) ++++++++++++++++++++++++++++++++++++++++++++++++++++#


def f_obj(peso):
    return np.sqrt(np.dot(peso.T, np.dot(sigma,peso)))


x0 = np.array([1.0/ (len(list)) for x in range(len(list))])

bounds = tuple((0, 1) for x in range (len(list)))

faixa_ret = np.arange(0.001, 2.5, 0.001)

risk = []

for i in faixa_ret:
    constrains = [{'type' : 'eq', 'fun' : lambda x: np.sum(x) - 1}, {'type' : 'eq', 'fun' : lambda x: np.sum(x * mi) - i}]
    outcome = solver.minimize(f_obj, x0, constraints=constrains,bounds=bounds , method='SLSQP')

    risk.append(outcome.fun)

fig.plot(risk, faixa_ret, 'r--x', linewidth = 5)

print('')
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('                 GOLL4.SA CVCB3.SA WEGE3.SA MGLU3.SA TOTS3.SA BOVA11.SA')
print('peso ótimo(w) = ', outcome['x'].round(9)) #numero 6 é referente ao aerredondamento de 6 casas decimais
print('+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++')
print('')

#fig.show()

#+++++++++++++++++++++++++++++++++++++++++++++++++++++( PONTO ÓTIMO - MÍNIMA VOLATILIDADE) ++++++++++++++++++++++++++++++++++++++++++++++++++++#


def estatistica_port(peso):
    peso = np.array(peso)
    ret_ot = np.sum(peso*mi)
    risco_ot = np.sqrt(np.dot(peso.T, np.dot(sigma,peso)))
    return np.array([ret_ot, risco_ot])
for i in faixa_ret:
    constrains = [{'type' : 'eq', 'fun' : lambda x: sum(x) - 1}]
    outcome = solver.minimize(f_obj, x0, constraints=constrains, bounds=bounds, method='SLSQP')

    risk.append(outcome.fun)

ret_ot, vol_ot = estatistica_port(outcome['x'])
print('retorno otimo esperado = ', str((ret_ot*100).round(len(list))) + '%')
print('volatilidade otima esperada = ', str((vol_ot*100).round(len(list))) + '%')
fig.plot(vol_ot, ret_ot, '*', markersize=35, markerfacecolor = 'w', markeredgecolor ='black')
fig.show()