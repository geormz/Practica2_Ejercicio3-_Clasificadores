from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score,mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import pandas as pd

# Cargar los datos de Pima
pima_data = pd.read_csv('pima.csv')
X_pima = pima_data.drop('Outcome', axis=1)
y_pima = pima_data['Outcome']

# Dividir los datos en conjuntos de entrenamiento y prueba para Pima
X_train_pima, X_test_pima, y_train_pima, y_test_pima = train_test_split(X_pima, y_pima, test_size=0.2, random_state=42)

# Entrenar el modelo de bosque aleatorio para Pima
clf_pima = RandomForestClassifier(random_state=42)
clf_pima.fit(X_train_pima, y_train_pima)
y_pred_pima = clf_pima.predict(X_test_pima)

# Calcular métricas para Pima
accuracy_pima = accuracy_score(y_test_pima, y_pred_pima)
precision_pima = precision_score(y_test_pima, y_pred_pima)
recall_pima = recall_score(y_test_pima, y_pred_pima)
conf_matrix_pima = confusion_matrix(y_test_pima, y_pred_pima)
f1_pima = f1_score(y_test_pima, y_pred_pima)

# Imprimir resultados para Pima
print("Métricas para Pima:")
print("Accuracy:", accuracy_pima)
print("Precision:", precision_pima)
print("Recall (Sensitivity):", recall_pima)
print("F1 Score:", f1_pima)
print("Confusion Matrix:")
print(conf_matrix_pima)
print()

# Entrenar el modelo KNN para Pima
knn_pima = KNeighborsClassifier()
knn_pima.fit(X_train_pima, y_train_pima)
y_pred_knn_pima = knn_pima.predict(X_test_pima)

# Calcular métricas para KNN y Pima
accuracy_knn_pima = accuracy_score(y_test_pima, y_pred_knn_pima)
precision_knn_pima = precision_score(y_test_pima, y_pred_knn_pima)
recall_knn_pima = recall_score(y_test_pima, y_pred_knn_pima)
conf_matrix_knn_pima = confusion_matrix(y_test_pima, y_pred_knn_pima)
f1_knn_pima = f1_score(y_test_pima, y_pred_knn_pima)

# Imprimir resultados para KNN y Pima
print("Métricas para KNN y Pima:")
print("Accuracy:", accuracy_knn_pima)
print("Precision:", precision_knn_pima)
print("Recall (Sensitivity):", recall_knn_pima)
print("F1 Score:", f1_knn_pima)
print("Confusion Matrix:")
print(conf_matrix_knn_pima)
print()

# Entrenar el modelo SVM para Pima
svm_pima = SVC(random_state=42)
svm_pima.fit(X_train_pima, y_train_pima)
y_pred_svm_pima = svm_pima.predict(X_test_pima)

# Calcular métricas para SVM y Pima
accuracy_svm_pima = accuracy_score(y_test_pima, y_pred_svm_pima)
precision_svm_pima = precision_score(y_test_pima, y_pred_svm_pima)
recall_svm_pima = recall_score(y_test_pima, y_pred_svm_pima)
conf_matrix_svm_pima = confusion_matrix(y_test_pima, y_pred_svm_pima)
f1_svm_pima = f1_score(y_test_pima, y_pred_svm_pima)

# Imprimir resultados para SVM y Pima
print("Métricas para SVM y Pima:")
print("Accuracy:", accuracy_svm_pima)
print("Precision:", precision_svm_pima)
print("Recall (Sensitivity):", recall_svm_pima)
print("F1 Score:", f1_svm_pima)
print("Confusion Matrix:")
print(conf_matrix_svm_pima)
print()

# Entrenar el modelo Naive Bayes para Pima
nb_pima = GaussianNB()
nb_pima.fit(X_train_pima, y_train_pima)
y_pred_nb_pima = nb_pima.predict(X_test_pima)

# Calcular métricas para Naive Bayes y Pima
accuracy_nb_pima = accuracy_score(y_test_pima, y_pred_nb_pima)
precision_nb_pima = precision_score(y_test_pima, y_pred_nb_pima)
recall_nb_pima = recall_score(y_test_pima, y_pred_nb_pima)
conf_matrix_nb_pima = confusion_matrix(y_test_pima, y_pred_nb_pima)
f1_nb_pima = f1_score(y_test_pima, y_pred_nb_pima)

# Imprimir resultados para Naive Bayes y Pima
print("Métricas para Naive Bayes y Pima:")
print("Accuracy:", accuracy_nb_pima)
print("Precision:", precision_nb_pima)
print("Recall (Sensitivity):", recall_nb_pima)
print("F1 Score:", f1_nb_pima)
print("Confusion Matrix:")
print(conf_matrix_nb_pima)
print()

