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

# Carregar o conjunto de dados
data = load_breast_cancer()
X = data.data
y = data.target

# Dividir o conjunto de dados em treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Algoritmo 1: SVM (Support Vector Machine)
svm_classifier = SVC(kernel='linear', C=1, gamma='scale', probability=True)

# Algoritmo 2: KNN (K-Nearest Neighbors)
knn_classifier = KNeighborsClassifier(n_neighbors=5)

# Algoritmo 3: Random Forest
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

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
        'tpr' : tpr
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
plt.legend()
plt.show()