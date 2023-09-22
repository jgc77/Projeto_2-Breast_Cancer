import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    auc,
    roc_auc_score,
)
from sklearn.tree import plot_tree
from sklearn.model_selection import learning_curve
from sklearn.preprocessing import StandardScaler
from mlxtend.plotting import plot_decision_regions

# Carregar o conjunto de dados
data = load_breast_cancer()
X = data.data
y = data.target

# Normalizar os dados
scaler = StandardScaler()
x_normalized = scaler.fit_transform(X)

df = pd.DataFrame(data.data, columns=data.feature_names)
df['target'] = data.target

# Matriz de Correlação
plt.figure(figsize=(20,20))
correlation_matrix = df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='mako', fmt=".0%")
plt.title('Matriz de Correlação')
plt.show()

#Grafico de disperção 
cols = ['target',
        'mean radius', 
        'mean texture', 
        'mean perimeter', 
        'mean area',
        'mean smoothness',
        'mean compactness',
        'mean concavity',
        'mean concave points',
        'mean symmetry',
        'mean fractal dimension']

sns.pairplot(data=df[cols], hue='target', palette='mako')

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(x_normalized, y, test_size=0.5, random_state=42)

# Algoritmo 1: SVM (Support Vector Machine)
svm_classifier = SVC(kernel='linear', C=1, gamma='scale', probability=True)

# Algoritmo 2: KNN (K-Nearest Neighbors)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Algoritmo 3: Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100,max_depth=4, random_state=42)

# Lista de algoritmos
classifiers = [('SVM', svm_classifier), ('KNN', knn_classifier), ('Random Forest', rf_classifier)]

# Realizar a validação cruzada e avaliar o desempenho dos modelos
results = []


for name, classifier in classifiers:
    # Realizar validação cruzada
    scores = cross_val_score(classifier, X_train, y_train, cv=5, scoring='accuracy')
    
    # Treinar o modelo no conjunto de treinamento completo
    classifier.fit(X_train, y_train)
    
    # Fazer previsões no conjunto de teste
    y_pred = classifier.predict(X_test)
    
    # Calcular métricas de desempenho
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Calcular a curva ROC e a área sob a curva (AUC)
    y_probs = classifier.predict_proba(X_test)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, y_probs)
    auc_score = auc(fpr, tpr)
    
    # Matriz de confusão
    cm = confusion_matrix(y_test, y_pred)

    # 
    if name == 'Random Forest' :
        # Plotando uma árvore de decisão individual da Floresta Aleatória
        plt.figure(figsize=(15, 10))
        plot_tree(classifier.estimators_[0], class_names=[str(i) for i in data.target_names], filled=True, rounded=True, fontsize=6)
        plt.show()
         # Calculando as curvas de aprendizado
        train_sizes, train_scores, test_scores = learning_curve(classifier, x_normalized, y, cv=5, scoring='accuracy', n_jobs=-1)

        # Calculando as médias e desvios padrão das pontuações em treinamento e teste
        train_scores_mean = np.mean(train_scores, axis=1)
        train_scores_std = np.std(train_scores, axis=1)
        test_scores_mean = np.mean(test_scores, axis=1)
        test_scores_std = np.std(test_scores, axis=1)

        # Plotando as curvas de aprendizado
        plt.figure(figsize=(10, 6))
        plt.title("Curva de Aprendizado (Learning Curve) RF")
        plt.xlabel("Tamanho do Conjunto de Treinamento")
        plt.ylabel("Precisão")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                        train_scores_mean + train_scores_std, alpha=0.1, color="r")
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                        test_scores_mean + test_scores_std, alpha=0.1, color="g")
        plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treinamento")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Teste")

        plt.legend(loc="best")
        plt.show()

        # Plotando Fronteiras de decisão
        # Criar um modelo de árvore de decisão que aceita apenas duas características
        #model_RandomForest = RandomForestClassifier(n_estimators=100,max_depth=2, random_state=42)

        # Ajustar o modelo aos dados de treinamento com apenas as duas características selecionadas
        classifier.fit(X_train[:, [27, 10]], y_train)
        # Use apenas as duas características selecionadas
        x_test_selected = X_test[:, [27, 10]]

        plot_decision_regions(x_test_selected, y_test, clf=classifier)
        plt.xlabel(data.feature_names[27])
        plt.ylabel(data.feature_names[10])
        plt.title('Fronteiras de Decisão')

        # Adicionando a legenda
        plt.legend(title="Breast Cancer")
        plt.show()

        
      
    if name == 'SVM' :
      # Calculando as curvas de aprendizado
      train_sizes, train_scores, test_scores = learning_curve(classifier, x_normalized, y, cv=5, scoring='accuracy', n_jobs=-1)

      # Calculando as médias e desvios padrão das pontuações em treinamento e teste
      train_scores_mean = np.mean(train_scores, axis=1)
      train_scores_std = np.std(train_scores, axis=1)
      test_scores_mean = np.mean(test_scores, axis=1)
      test_scores_std = np.std(test_scores, axis=1)

      # Plotando as curvas de aprendizado
      plt.figure(figsize=(10, 6))
      plt.title("Curva de Aprendizado (Learning Curve) SVM")
      plt.xlabel("Tamanho do Conjunto de Treinamento")
      plt.ylabel("Precisão")
      plt.grid()

      plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
      plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
      plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treinamento")
      plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Teste")

      plt.legend(loc="best")
      plt.show()
      # Plotando Fronteiras de decisão
      # Criar um modelo de SVM que aceita apenas duas características
      #model_SVM = SVC(kernel='linear', C=1, gamma='scale', probability=True)

      # Ajustar o modelo aos dados de treinamento com apenas as duas características selecionadas
      classifier.fit(X_train[:, [27, 10]], y_train)

      # Use apenas as duas características selecionadas
      x_test_selected = X_test[:, [27, 10]]

      plot_decision_regions(x_test_selected, y_test, clf=classifier)
      plt.xlabel(data.feature_names[27])
      plt.ylabel(data.feature_names[10])
      plt.title('Fronteiras de Decisão')

      # Adicionando a legenda
      plt.legend(title="Breast Cancer")
      plt.show()

    if name == 'KNN' :
      # Calculando as curvas de aprendizado
      train_sizes, train_scores, test_scores = learning_curve(classifier, x_normalized, y, cv=5, scoring='accuracy', n_jobs=-1)

      # Calculando as médias e desvios padrão das pontuações em treinamento e teste
      train_scores_mean = np.mean(train_scores, axis=1)
      train_scores_std = np.std(train_scores, axis=1)
      test_scores_mean = np.mean(test_scores, axis=1)
      test_scores_std = np.std(test_scores, axis=1)

      # Plotando as curvas de aprendizado
      plt.figure(figsize=(10, 6))
      plt.title("Curva de Aprendizado (Learning Curve) KNN")
      plt.xlabel("Tamanho do Conjunto de Treinamento")
      plt.ylabel("Precisão")
      plt.grid()

      plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                      train_scores_mean + train_scores_std, alpha=0.1, color="r")
      plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                      test_scores_mean + test_scores_std, alpha=0.1, color="g")
      plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Treinamento")
      plt.plot(train_sizes, test_scores_mean, 'o-', color="g", label="Teste")

      plt.legend(loc="best")
      plt.show()

      # Plotando Fronteiras de decisão
      # Criar um modelo de SVM que aceita apenas duas características
      #model_KNN = SVC(kernel='linear', C=1, gamma='scale', probability=True)

      # Ajustar o modelo aos dados de treinamento com apenas as duas características selecionadas
      classifier.fit(X_train[:, [27, 10]], y_train)

      # Use apenas as duas características selecionadas
      x_test_selected = X_test[:, [27, 10]]

      plot_decision_regions(x_test_selected, y_test, clf=classifier)
      plt.xlabel(data.feature_names[27])
      plt.ylabel(data.feature_names[10])
      plt.title('Fronteiras de Decisão')

      # Adicionando a legenda
      plt.legend(title="Breast Cancer")
      plt.show()
    
    results.append({
        'Algoritmo': name,
        'Acurácia (CV)': np.mean(scores),
        'Acurácia': accuracy,
        'Precisão': precision,
        'Recall': recall,
        'F1 Score': f1,
        'AUC': auc_score,
        'Matriz de Confusão': cm,
        'fpr' : fpr,
        'tpr' : tpr,
    })
  
# Exibir os resultados
for result in results:
    print(f"Algoritmo: {result['Algoritmo']}")
    print(f"Acurácia (CV): {result['Acurácia (CV)']:.2f}")
    print(f"Acurácia: {result['Acurácia']:.2f}")
    print(f"Precisão: {result['Precisão']:.2f}")
    print(f"Recall: {result['Recall']:.2f}")
    print(f"F1 Score: {result['F1 Score']:.2f}")
    print(f"AUC: {result['AUC']:.2f}")
    print(f"Matriz de Confusão:\n{result['Matriz de Confusão']}\n")

    # Plotar a curva ROC para cada algoritmo
    plt.figure(figsize=(8, 6))
    plt.plot(result['fpr'], result['tpr'], color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve ({result["Algoritmo"]})')
    plt.legend(loc='lower right')
    plt.show()

    # Plotar a matriz de confusão para cada algoritmo
    plt.figure(figsize=(6, 6))
    sns.heatmap(result['Matriz de Confusão'], annot=True, fmt='d', cmap='Blues', cbar=False, square=True)
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.title(f'Confusion Matrix ({result["Algoritmo"]})')
    plt.show()

    

# Gráfico de comparação de desempenho
plt.figure(figsize=(10, 6))
for result in results:
    plt.bar(result['Algoritmo'], result['Acurácia (CV)'], label=result['Algoritmo'])
plt.xlabel('Algoritmo')
plt.ylabel('Acurácia (CV)')
plt.title('Comparação de Desempenho de Algoritmos')
plt.grid()
plt.legend()
plt.show()
