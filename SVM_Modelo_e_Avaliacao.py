import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, roc_auc_score, roc_curve
from scipy.stats import ttest_rel
import pywt
from scipy.fft import fft, fftfreq
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import matthews_corrcoef
from sklearn.inspection import permutation_importance

# Importação da base de dados
bd = pd.read_csv('vibration.csv')
print("Primeiras linhas do dataset:")
print(bd.head())
print(f"Número total de linhas: {len(bd)}")

# Parâmetro: comprimento de cada sinal (5s a 1000 Hz = 5000 amostras)
signal_length = 5000
n_signals = len(bd) // signal_length
print(f"Número estimado de sinais: {n_signals}")

# Ajuste do dataset se não for divisível
if len(bd) % signal_length != 0:
    print(f"Aviso: O dataset não é divisível em sinais de {signal_length} amostras. Ajustando...")
    bd = bd.iloc[:n_signals * signal_length]

# Separação dos sinais
x = bd.iloc[:, 0:3].values  # Sinais triaxiais (x, y, z)
y = bd.iloc[:, 3].values    # Rótulos (0 = normal, 1 = falha)

# Combinação dos eixos x, y, z
x_combined = np.sqrt(np.sum(x ** 2, axis=1))
signals = x_combined.reshape(n_signals, signal_length)
labels = y[::signal_length]  # Rótulo constante por sinal

# Divisão em treino e teste
train_signals, test_signals, train_y, test_y = train_test_split(signals, labels, test_size=0.2, random_state=42)

# Normalização dos sinais
scaler = StandardScaler()
train_signals_scaled = np.array([scaler.fit_transform(sig.reshape(-1, 1)).flatten() for sig in train_signals])
test_signals_scaled = np.array([scaler.transform(sig.reshape(-1, 1)).flatten() for sig in test_signals])

# Função para aplicar Transformada Wavelet
def apply_wavelet(signal):
    coeffs = pywt.wavedec(signal, 'db4', level=4)
    coeffs_mod = [np.zeros_like(coeffs[0])] + coeffs[1:]  # Zera a aproximação
    return pywt.waverec(coeffs_mod, 'db4')

# Aplicando Wavelet
train_signals_wavelet = np.array([apply_wavelet(sig) for sig in train_signals_scaled])
test_signals_wavelet = np.array([apply_wavelet(sig) for sig in test_signals_scaled])

# Extração de características temporais
def extract_temporal_features(signal):
    features = {
        'mean': np.mean(signal),
        'std': np.std(signal),
        'kurtosis': np.mean((signal - np.mean(signal)) ** 4) / (np.std(signal) ** 4 + 1e-10),
        'skewness': np.mean((signal - np.mean(signal)) ** 3) / (np.std(signal) ** 3 + 1e-10),
        'rms': np.sqrt(np.mean(signal ** 2)),
        'peak_to_peak': np.max(signal) - np.min(signal),
        'energy': np.sum(signal ** 2)
    }
    return list(features.values())

# Extração de características espectrais
def extract_spectral_features(signal):
    fft_signal = fft(signal)
    mag_spectrum = np.abs(fft_signal)
    freqs = fftfreq(len(signal))
    pos_mask = freqs > 0
    freqs = freqs[pos_mask]
    mag_spectrum = mag_spectrum[pos_mask]

    features = {
        'mean_freq': np.sum(freqs * mag_spectrum) / (np.sum(mag_spectrum) + 1e-10),
        'std_freq': np.sqrt(np.sum(mag_spectrum * (freqs - np.sum(freqs * mag_spectrum) / (np.sum(mag_spectrum) + 1e-10)) ** 2) / (np.sum(mag_spectrum) + 1e-10)),
        'spectral_kurtosis': np.mean((mag_spectrum - np.mean(mag_spectrum)) ** 4) / (np.std(mag_spectrum) ** 4 + 1e-10),
        'spectral_skewness': np.mean((mag_spectrum - np.mean(mag_spectrum)) ** 3) / (np.std(mag_spectrum) ** 3 + 1e-10),
        'band_power': np.sum(mag_spectrum ** 2) / len(mag_spectrum),
        'median_freq': freqs[np.argmin(np.abs(np.cumsum(mag_spectrum) - np.sum(mag_spectrum) / 2))]
    }
    return list(features.values())

# Criando vetores de características híbridas
train_features = np.array([extract_temporal_features(sig) + extract_spectral_features(sig) for sig in train_signals_wavelet])
test_features = np.array([extract_temporal_features(sig) + extract_spectral_features(sig) for sig in test_signals_wavelet])

# Seleção de atributos por correlação
all_feature_names = ['mean', 'std', 'kurtosis', 'skewness', 'rms', 'peak_to_peak', 'energy',
                    'mean_freq', 'std_freq', 'spectral_kurtosis', 'spectral_skewness', 'band_power', 'median_freq']
correlation_matrix = np.corrcoef(train_features.T)
highly_correlated = np.where(np.abs(correlation_matrix) > 0.9)
to_remove = set()
for i, j in zip(highly_correlated[0], highly_correlated[1]):
    if i != j and i < j:
        to_remove.add(j)
selected_features = np.delete(train_features, list(to_remove), axis=1)
test_features_selected = np.delete(test_features, list(to_remove), axis=1)
removed_features = [all_feature_names[i] for i in sorted(to_remove)]
print(f"Features removidas por correlação alta: {removed_features}")
print(f"Features restantes: {len(selected_features[0])}")
print(f"Novas dimensões das features: {selected_features.shape[1]}")

# Normalização das features selecionadas
scaler_features = StandardScaler()
train_features_scaled = scaler_features.fit_transform(selected_features)
test_features_scaled = scaler_features.transform(test_features_selected)

# Definição dos classificadores
classifiers = {
    'SVM (Linear Kernel)': SVC(kernel='linear', class_weight='balanced', random_state=42),
    'SVM (Quadratic Kernel)': SVC(kernel='poly', degree=2, class_weight='balanced', random_state=42),
    'SVM (Cubic Kernel)': SVC(kernel='poly', degree=3, class_weight='balanced', random_state=42),
    'SVM (Gaussian Kernel)': SVC(kernel='rbf', class_weight='balanced', random_state=42),
    'KNN (k=1)': KNeighborsClassifier(n_neighbors=1),
    'KNN (k=5)': KNeighborsClassifier(n_neighbors=5),
    'KNN (Weighted)': KNeighborsClassifier(n_neighbors=5, weights='distance'),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'LDA': LinearDiscriminantAnalysis()
}

# Grid Search para otimização do SVM
param_grid = {
    'SVM (Linear Kernel)': {'C': [0.1, 1, 10]},
    'SVM (Quadratic Kernel)': {'C': [0.1, 1, 10], 'degree': [2, 3]},
    'SVM (Cubic Kernel)': {'C': [0.1, 1, 10], 'degree': [3, 4]},
    'SVM (Gaussian Kernel)': {'C': [0.1, 1, 10], 'gamma': ['scale', 'auto', 0.1]}
}
optimized_classifiers = {}
for name, clf in classifiers.items():
    if 'SVM' in name:
        grid_search = GridSearchCV(clf, param_grid[name], cv=5, scoring='accuracy')
        grid_search.fit(train_features_scaled, train_y)
        optimized_classifiers[name] = grid_search.best_estimator_
        print(f'Melhor {name}: {grid_search.best_params_}, Score: {grid_search.best_score_:.4f}')
    else:
        optimized_classifiers[name] = clf.fit(train_features_scaled, train_y)

# Validação cruzada
cv_scores = {}
for name, clf in optimized_classifiers.items():
    scores = cross_val_score(clf, train_features_scaled, train_y, cv=10)
    cv_scores[name] = scores
    print(f'{name}: Accuracy = {np.mean(scores):.4f} ± {np.std(scores):.4f}')

# Métricas no conjunto de teste
metrics = {}
roc_data = {}
for name, clf in optimized_classifiers.items():
    clf.fit(train_features_scaled, train_y)
    predictions = clf.predict(test_features_scaled)

    accuracy = accuracy_score(test_y, predictions)
    conf_matrix = confusion_matrix(test_y, predictions)
    sensitivity = conf_matrix[1, 1] / (conf_matrix[1, 1] + conf_matrix[1, 0]) if (conf_matrix[1, 1] + conf_matrix[1, 0]) > 0 else 0
    specificity = conf_matrix[0, 0] / (conf_matrix[0, 0] + conf_matrix[0, 1]) if (conf_matrix[0, 0] + conf_matrix[0, 1]) > 0 else 0
    f1 = f1_score(test_y, predictions, average='binary')

    if hasattr(clf, 'decision_function'):
        y_scores = clf.decision_function(test_features_scaled)
    elif hasattr(clf, 'predict_proba'):
        y_scores = clf.predict_proba(test_features_scaled)[:, 1]
    else:
        y_scores = predictions
    auc = roc_auc_score(test_y, y_scores)
    fpr, tpr, _ = roc_curve(test_y, y_scores)

    metrics[name] = {'Accuracy': accuracy, 'Sensitivity': sensitivity, 'Specificity': specificity, 'F1-Score': f1, 'AUC-ROC': auc}
    roc_data[name] = (fpr, tpr)

# Melhor modelo
best_model_name = max(metrics, key=lambda x: metrics[x]['Accuracy'])
best_model = optimized_classifiers[best_model_name]
best_model.fit(train_features_scaled, train_y)
best_predictions = best_model.predict(test_features_scaled)

best_accuracy = accuracy_score(test_y, best_predictions)
best_conf_matrix = confusion_matrix(test_y, best_predictions)
best_sensitivity = best_conf_matrix[1, 1] / (best_conf_matrix[1, 1] + best_conf_matrix[1, 0]) if (best_conf_matrix[1, 1] + best_conf_matrix[1, 0]) > 0 else 0
best_specificity = best_conf_matrix[0, 0] / (best_conf_matrix[0, 0] + best_conf_matrix[0, 1]) if (best_conf_matrix[0, 0] + best_conf_matrix[0, 1]) > 0 else 0
best_f1 = f1_score(test_y, best_predictions, average='binary')
best_y_scores = best_model.decision_function(test_features_scaled) if hasattr(best_model, 'decision_function') else best_model.predict_proba(test_features_scaled)[:, 1]
best_auc = roc_auc_score(test_y, best_y_scores)
best_fpr, best_tpr, _ = roc_curve(test_y, best_y_scores)

metrics[best_model_name] = {'Accuracy': best_accuracy, 'Sensitivity': best_sensitivity, 'Specificity': best_specificity, 'F1-Score': best_f1, 'AUC-ROC': best_auc}
roc_data[best_model_name] = (best_fpr, best_tpr)

# Tabela de métricas
metrics_df = pd.DataFrame(metrics).T
print("\nMétricas para todos os classificadores no conjunto de teste:")
print(metrics_df)

# Teste t pareado
svm_quad_scores = cv_scores.get('SVM (Quadratic Kernel)', [])
p_values = {}
for name, scores in cv_scores.items():
    if name != 'SVM (Quadratic Kernel)' and len(svm_quad_scores) > 0:
        t_stat, p_value = ttest_rel(svm_quad_scores, scores)
        p_values[name] = p_value
print("\nP-valores do teste t pareado (SVM Quadrático vs. outros):")
for name, p in p_values.items():
    print(f'{name}: p-value = {p:.4f}')

# Curvas ROC
plt.figure(figsize=(8, 6))
for name, (fpr, tpr) in roc_data.items():
    auc = metrics[name]['AUC-ROC']
    plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.2f})')
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel('Taxa de Falsos Positivos')
plt.ylabel('Taxa de Verdadeiros Positivos')
plt.title('Curvas ROC para Todos os Classificadores')
plt.legend()
plt.show()

# Matriz de confusão do melhor modelo
plt.figure(figsize=(6, 4))
sns.heatmap(best_conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Normal', 'Faulty'], yticklabels=['Normal', 'Faulty'])
plt.title(f'Matriz de Confusão - {best_model_name}')
plt.xlabel('Previsto')
plt.ylabel('Real')
plt.show()

# Análise de Importância das Variáveis
remaining_feature_names = [name for idx, name in enumerate(all_feature_names) if idx not in sorted(to_remove)]
perm_importance = permutation_importance(best_model, test_features_scaled, test_y, n_repeats=10, random_state=42)
feature_importance = perm_importance.importances_mean
sorted_idx = np.argsort(feature_importance)[::-1]
feature_importance_sorted = feature_importance[sorted_idx]
feature_names_sorted = [remaining_feature_names[i] for i in sorted_idx]

print("\nFeatures removidas por correlação alta:", removed_features)
print("\nImportância das Features restantes (ordem decrescente):")
for name, importance in zip(feature_names_sorted, feature_importance_sorted):
    print(f"{name}: {importance:.4f}")

plt.figure(figsize=(8, 6))
plt.bar(range(len(feature_importance_sorted)), feature_importance_sorted)
plt.xticks(range(len(feature_importance_sorted)), feature_names_sorted, rotation=45)
plt.xlabel('Features')
plt.ylabel('Importância Média')
plt.title('Importância das Features no SVM Linear')
plt.tight_layout()
plt.savefig('Importancia_das_features.png')
plt.show()

# ----------------------------------------
# Visualização 2D do SVM Linear
# ----------------------------------------

# Escolhe as duas features mais importantes
top2_idx = sorted_idx[:2]
X_vis = test_features_scaled[:, top2_idx]
y_vis = test_y

# Treina um novo SVM só com essas duas features
svm_vis = SVC(kernel='linear', class_weight='balanced', random_state=42)
svm_vis.fit(X_vis, y_vis)

# Cria o grid para visualizar a fronteira de decisão
x_min, x_max = X_vis[:, 0].min() - 1, X_vis[:, 0].max() + 1
y_min, y_max = X_vis[:, 1].min() - 1, X_vis[:, 1].max() + 1
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 300),
                     np.linspace(y_min, y_max, 300))

Z = svm_vis.decision_function(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, levels=20, cmap='coolwarm', alpha=0.3)
plt.scatter(X_vis[y_vis == 0, 0], X_vis[y_vis == 0, 1],
            color='blue', label='Normal (0)', edgecolor='k')
plt.scatter(X_vis[y_vis == 1, 0], X_vis[y_vis == 1, 1],
            color='red', label='Falha (1)', edgecolor='k')
plt.xlabel(feature_names_sorted[0])
plt.ylabel(feature_names_sorted[1])
plt.title('Fronteira de Decisão do SVM (2D)')
plt.legend()
plt.tight_layout()
plt.savefig('fronteira_de_decisao.png')
plt.show()

from mpl_toolkits.mplot3d import Axes3D

top3_idx = sorted_idx[:3]
X3 = test_features_scaled[:, top3_idx]
y3 = test_y

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X3[y3 == 0, 0], X3[y3 == 0, 1], X3[y3 == 0, 2], color='blue', label='Normal')
ax.scatter(X3[y3 == 1, 0], X3[y3 == 1, 1], X3[y3 == 1, 2], color='red', label='Falha')
ax.set_xlabel(feature_names_sorted[0])
ax.set_ylabel(feature_names_sorted[1])
ax.set_zlabel(feature_names_sorted[2])
ax.set_title('Espaço de Features - SVM (3D)')
ax.legend()
plt.tight_layout()
plt.savefig('fronteira_de_decisao _3d.png')
plt.show()