import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

#Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/jgc77/Projeto_2-Breast_Cancer/main/cancer.csv')
df.head()

#Tratamento de dados 
df = df.dropna(axis=1)
df = df.drop('id', axis=1)

#Quantidade de tumores malignos e benignos
#df['diagnosis'].value_counts()
qb = (df['diagnosis'] == 0).sum()
qm = (df['diagnosis'] == 1).sum()
print("Quantidade de tumores Benignos: ", qb)
print("Quantidade de tumores Malignos: ", qm)

#T
sns.countplot(x='diagnosis', data=df, palette='husl')
plt.xlabel('Diagnóstico')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Maligno', 'Benigno'])  # Rótulos para o eixo x
plt.show()

#Alterar B para 0 e M para 1 
diagnosis_mapping = {'B': 0, 'M': 1}
df['diagnosis'] = df['diagnosis'].map(diagnosis_mapping)

#Grafico de disperção com todas as colunas
cols = ['diagnosis',
        'radius_mean', 
        'texture_mean', 
        'perimeter_mean', 
        'area_mean', 
        'smoothness_mean', 
        'compactness_mean', 
        'concavity_mean',
        'concave points_mean', 
        'symmetry_mean', 
        'fractal_dimension_mean']

sns.pairplot(data=df[cols], hue='diagnosis', palette='mako')


#RandonFlorest

# Separar os dados em recursos (x1) e rótulos (y)
x1 = df.drop('diagnosis', axis=1)
y = df['diagnosis']

# Dividir os dados em conjunto de treinamento e teste
x_train, x_test, y_train, y_test = train_test_split(x1, y, test_size=0.2, random_state=21)

# Treinar o modelo Random Forest
clf_rf = RandomForestClassifier(random_state=42)
clr_rf = clf_rf.fit(x_train, y_train)

# Calcular a acurácia do modelo no conjunto de teste
ac = accuracy_score(y_test, clf_rf.predict(x_test))
print('Accuracy is:', ac * 100)

# Calcular a quantidade de tumores benignos no conjunto de teste
quantidade_benignos_teste = (y_test == 0).sum()
print(f"A quantidade de tumores benignos no conjunto de teste é: {quantidade_benignos_teste}")

# Exibir a matriz de confusão
cm = confusion_matrix(y_test, clf_rf.predict(x_test))
sns.heatmap(cm / np.sum(cm), annot=True, fmt=".2%", cmap='Blues')
plt.show()





