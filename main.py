import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

file_path = 'housing_price_dataset.csv'

df = pd.read_csv(file_path)

# Criando uma coluna numérica para representar 'Price'
df['PriceNumeric'] = pd.cut(df['Price'], bins=[0, 250000, 300000, float('inf')], labels=[1, 2, 3])

# Removendo linhas que contêm valores ausentes
df = df.dropna()

# Separando features (X) e rótulos (y)
X = df[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt']]
y = df['PriceNumeric']

# Convertendo a coluna categórica
X_encoded = pd.get_dummies(X, columns=['Neighborhood'], drop_first=True)

# Dividindo o conjunto de dados
X_treino, X_teste, y_treino, y_teste = train_test_split(X_encoded, y, test_size=0.2, random_state=3)

# Inicializando o classificador k-NN
classificador_knn = KNeighborsClassifier(n_neighbors=6)

# Treinando o classificador
classificador_knn.fit(X_treino, y_treino)

# Fazendo previsões com os dados
y_predito = classificador_knn.predict(X_teste)

# Avaliando a precisão
precisao = accuracy_score(y_teste, y_predito)

print(f'A precisão do modelo é: {precisao * 100:.2f}%')

# Média de preços geral
media_precos_geral = df['PriceNumeric'].astype(float).mean()
print(f'Média de preços geral: {media_precos_geral}')

# Diferença de preços entre casas em área urbana e suburbana
precos_urbana = df[df['Neighborhood'] == 'Urban']['PriceNumeric'].astype(float).mean()
precos_suburbana = df[df['Neighborhood'] == 'Suburb']['PriceNumeric'].astype(float).mean()
diferenca_precos = precos_urbana - precos_suburbana
print(f'Diferença de preços entre área urbana e suburbana: {diferenca_precos}')

# Casa com o melhor preço por espaço
df['PrecoPorEspaco'] = df['PriceNumeric'].astype(float) / df['SquareFeet']
melhor_casa = df.loc[df['PrecoPorEspaco'].idxmax()]
print(f'Melhor casa em termos de preço por espaço:')
print(melhor_casa[['SquareFeet', 'Bedrooms', 'Bathrooms', 'Neighborhood', 'YearBuilt', 'Price', 'PrecoPorEspaco']])
