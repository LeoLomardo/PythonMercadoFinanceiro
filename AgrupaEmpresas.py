import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import cross_val_score, KFold
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV
import pickle

dataset = pd.read_excel('Dados.xlsx')
pd.options.display.max_columns = None
pd.options.display.max_rows = None
print(dataset.shape)

dataset.drop(['Número de Empregados','Número de Acionistas', 'CAGR RECEITAS 5 ANOS', 'CAGR LUCROS 5 ANOS', 'CÓDIGO', 'PATRIMONIO / ATIVOS', 'Valor da Empresa (Trim)'], axis=1, inplace=True)
print(dataset.isnull().sum())  # aqui ele mostra quandos valores faltantes tem de cada coluna
dataset['Número de Acionistas'].fillna(dataset['Número de Acionistas'].mean(), inplace=True)  # preenche o resto dos valores faltantes com a media dos valores da coluna
dataset['DY'].fillna(0, inplace=True)  # preenche o resto dos valores faltantes com a media dos valores da coluna
dataset['Dividendos por Ação (AF)'].fillna(0, inplace=True)
dataset['DIV. LIQ. / PATRI.'].fillna(0, inplace=True)
dataset['ROIC'].fillna(0, inplace=True)
dataset['P/CAP. GIRO'].fillna(0, inplace=True)

# dataset[''].fillna(0, inplace=True)
sns.heatmap(dataset.isnull())

# print('++++++++++++++++++++++++++++ ( REMOÇÃO DE COLUNAS COM GRANDE QUANTIDADE DE DADOS FALTANTES ) ++++++++++++++++++++++++++++++++++')
print('++++++++++++++++++++++++++++ ( QUANTIDADE FALTANTE DE DADOS ) ++++++++++++++++++++++++++++++++++')
dataset.drop(labels=['PSR', 'MARGEM EBIT', 'Total de Ativos (Trim)', 'PEG Ratio', 'DIVIDA LIQUIDA / EBIT', 'MARG. LIQUIDA', 'EBITDA (12M)', 'LIQ. CORRENTE', 'P/EBIT'], axis=1, inplace=True)  # retira dados iniciais
print(dataset.isnull().sum())  # aqui ele mostra quandos valores faltantes tem de cada coluna


print('++++++++++++++++++++++++++++ (TABELA APÓS PREENCHEMENTO DADOS FALTANTES ) ++++++++++++++++++++++++++++++++++')
dataset.fillna(dataset.mean(), inplace=True)  # preenche o resto dos valores faltantes com a media dos valores da coluna
pd.options.display.max_columns = None
print(dataset.describe())
# dataset.fillna([], inplace=True)
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ( VISUALIZAÇÃO DISPOSIÇÃO DOS DADOS ) ++++++++++++++++++++++++++++++++++++++++ #
figura = plt.figure(figsize=(30, 15))
eixo = figura.gca()
dataset.hist(ax=eixo)
dataset.drop([], axis=1, inplace=True)  # retirei as colunas que estavam com muitos valores faltantes
print(dataset.isnull().sum())  # aqui ele mostra quandos valores faltantes tem de cada coluna
plt.figure(figsize=(30, 50))
sns.heatmap(dataset.corr(), annot=True, cbar=False)

plt.show()

print('Tendências: ', np.unique(dataset['AVALIAÇÃO'], return_counts=True))
sns.countplot(x=dataset['AVALIAÇÃO'])
plt.show()
print('Setores: ', np.unique(dataset['Setor'], return_counts=True))
sns.countplot(x=dataset['Setor'])
plt.show()
print('Indústrias: ', np.unique(dataset['Indústria'], return_counts=True))
sns.countplot(x=dataset['Indústria'])
plt.show()


# # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ( VARIÁVEIS DUMMY ) ++++++++++++++++++++++++++++++++++++++++ #
empresa = dataset['Descrição']
y = dataset['AVALIAÇÃO'].values
X_cat = dataset[['Indústria', 'Setor']]
print('Dimensão X_cat:', X_cat.shape)

onehotencoder = OneHotEncoder()
X_cat = onehotencoder.fit_transform(X_cat).toarray()
print(X_cat)


X_cat = pd.DataFrame(X_cat)
dataset_original = dataset.copy()

dataset.drop(['AVALIAÇÃO', 'Setor', 'Indústria', 'Descrição'], axis=1, inplace=True)
print(dataset)

dataset.index = X_cat.index
dataset = pd.concat([dataset, X_cat], axis=1)
print(dataset)

# # # # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ( NORMALIZAÇÃO ) ++++++++++++++++++++++++++++++++++++++++ #
scaler = MinMaxScaler()
dataset_normalizado = scaler.fit_transform(dataset)
X = dataset_normalizado.copy()
print(y)
print('Dimensoes X: ', X.shape)
# # # # # # ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ( APLICAÇÃO ALGORÍTMO ) ++++++++++++++++++++++++++++++++++++++++ #


resultados_forest = []
resultados_neural = []
for i in range(50):  # numero de testes
    kfold = KFold(n_splits=10, shuffle=True, random_state=i)  # 10 desmembramentos, cenário científico mantém 10 como padrão

    random_forest = RandomForestClassifier()
    scores = cross_val_score(random_forest, X, y, cv=kfold)
    resultados_forest.append(scores.mean())

    network = MLPClassifier(hidden_layer_sizes=(211, 211))
    scores = cross_val_score(network, X, y, cv=kfold)
    resultados_neural.append(scores.mean())

resultados_forest = np.array(resultados_forest)
resultados_neural = np.array(resultados_neural)

print('Resultado Algoritmo Forest: ', resultados_forest)
print('Resultado Algoritmo Neural: ', resultados_neural)
print('Média Algoritmo Forest: ', resultados_forest.mean())
print('Média Algoritmo Neural: ', resultados_neural.mean())
#  ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ ( AVALIAÇÃO COM BASE DE TREINAMENTO E TESTE ) ++++++++++++++++++++++++++++++++++++++++ #

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y, test_size=0.3)  # 0.3 referente a 30% da base de dados para teste
print('Tamanho X treinamento: ', X_treinamento.shape)
print('Tamanho Y teste',  y_treinamento.shape)

random_forest = RandomForestClassifier()
random_forest.fit(X_treinamento, y_treinamento)

previsoes = random_forest.predict(X_teste)
print('Previsões: ', previsoes)
print('Y teste:', y_teste)
cm = confusion_matrix(y_teste, previsoes)

sns.heatmap(cm, annot=True)
print(classification_report(y_teste, previsoes))
X_teste[0].reshape(1, -1)

random_forest.predict(X_teste[0].reshape(1, -1))
print('random_forest.feature_importances_')
np.argmax(random_forest.feature_importances_)

for nome, importancia in zip(dataset.columns, random_forest.feature_importances_):
    print(nome, ' = ', importancia)

caracteristicas = dataset.columns
importancias = random_forest.feature_importances_
indices = np.argsort(importancias)

plt.figure(figsize=(40, 50))
plt.title('Importância das características')
plt.barh(range(len(indices)), importancias[indices], color='b', align='center')
plt.yticks(range(len(indices)), [caracteristicas[i] for i in indices])
plt.xlabel('Importâncias')
plt.show()

with open('base_definitiva.pkl', 'wb') as f:
    pickle.dump([dataset, dataset_original, X, y, scaler], f)

parametros = {'criterion': ['gini', 'entropy'], 'min_samples_split': [2, 4, 6], 'n_estimators': [50, 100, 150]}

grid_search = GridSearchCV(estimator=RandomForestClassifier(), param_grid=parametros)
grid_search.fit(X, y)
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print(best_params, best_score)
