import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/jgc77/Projeto_2-Breast_Cancer/main/cancer.csv')
df.head()

#Tratamento de dados 
df = df.dropna(axis=1)
df['diagnosis'].value_counts()

#Plotar quantidade de cancer maligno e benigno 
sns.countplot(x='diagnosis', data=df, palette='husl')
plt.xlabel('Diagnóstico')
plt.ylabel('Contagem')
plt.xticks([0, 1], ['Maligno', 'Benigno'])  # Rótulos para o eixo x
plt.show()

#Alterar B para 0 e M para 1 
diagnosis_mapping = {'B': 0, 'M': 1}
df['diagnosis'] = df['diagnosis'].map(diagnosis_mapping)







