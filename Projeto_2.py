import pandas as pd
import numpy as np

#Carregar dataset
df = pd.read_csv('https://raw.githubusercontent.com/jgc77/Projeto_2-Breast_Cancer/main/cancer.csv')
df.head()

#Tratamento de dados 
df = df.dropna(axis=1)
df['diagnosis'].value_counts()

#Alterar B para 0 e M para 1 
diagnosis_mapping = {'B': 0, 'M': 1}
df['diagnosis'] = df['diagnosis'].map(diagnosis_mapping)