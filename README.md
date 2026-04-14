🇧🇷 Versão em Português (README.md)

# ⚙️ Preditiva-IoT: Detecção de Falhas Industriais via Análise de Vibração

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Model-Scikit--Learn-F7931E?logo=scikit-learn)
![Scipy](https://img.shields.io/badge/Signal_Processing-Scipy-8CAAE6?logo=scipy)
![Pandas](https://img.shields.io/badge/Data_Engineering-Pandas-150458?logo=pandas)
![Industry 4.0](https://img.shields.io/badge/Domain-Industry_4.0-success)

Este repositório apresenta uma solução avançada de **Machine Learning para Manutenção Preditiva**, focada em detectar anomalias em máquinas rotativas utilizando sinais de vibração triaxiais. O projeto engloba desde a ingestão de ficheiros raw (MATLAB) até à extração de características no domínio do tempo e da frequência (Wavelets/FFT) e rigorosa avaliação estatística.

---

## 📌 Project Overview (Visão de Negócio)

* **Problem Statement:** Em chãos de fábrica e infraestruturas críticas, falhas em componentes mecânicos (como rolamentos e motores) resultam em paragens não planeadas, riscos de segurança e elevados custos de manutenção corretiva.
* **Value Proposition:** Ao analisar a assinatura vibracional dos equipamentos em tempo quase-real, esta solução permite identificar padrões microscópicos de degradação antes que a falha funcional ocorra. 
* **KPIs de Sucesso Impactados:**
  * Aumento do **OEE** (Overall Equipment Effectiveness) através da redução do tempo de inatividade.
  * Otimização dos custos de manutenção (redução de substituição prematura de peças).
  * Aumento do **MTBF** (Mean Time Between Failures).

---

## 🏗️ Architecture & Engineering

A pipeline foi construída com um forte foco em Processamento Digital de Sinais (DSP) e Estatística.

### 1. Data Pipeline (ETL)
* Um script dedicado extrai os dados brutos de sensores no formato `.mat` (matrizes MATLAB de séries temporais), converte-os em DataFrames relacionais e rotula os dados como saudáveis (`0`) ou em falha (`1`).

### 2. Signal Processing & Feature Engineering
* **Denoising:** Aplicação de **Transformada Wavelet Discreta (Daubechies 4)** para atenuar o ruído industrial sem perder os picos transientes característicos de fissuras ou atritos.
* **Domínio do Tempo:** Cálculo de métricas robustas (RMS, Curtose, Assimetria, Pico-a-Pico).
* **Domínio da Frequência:** Aplicação de **Fast Fourier Transform (FFT)** para extrair espectros de potência e frequências medianas.
* **Feature Selection:** Remoção de variáveis colineares (correlação de Pearson > 0.90) para prevenir *overfitting*.

### 3. Model Stack
* Diversas arquiteturas foram comparadas via `GridSearchCV` (K-NN, Decision Trees, LDA, SVMs com kernels variados).

---

## 🚀 Get Started (Quick Start)

**1. Gerar o Dataset Estruturado**
O script ETL irá ler os diretórios `archive/Healthy` e `archive/Faulty` e agregar tudo num único CSV.
```bash
python ETL.py
```

**2. Executar o Pipeline de Treino e Avaliação**
Este comando processará os sinais, fará a extração das features e avaliará os modelos, gerando gráficos de performance.
```bash
python SVM_Modelo_e_Avaliacao.py
```

---

## 📊 Model Performance & Engineering Excellence

* **Performance de Topo:** O classificador final (Support Vector Machine com Kernel Linear) alcançou uma **Acurácia de 97.73%** e um **Recall/Sensibilidade de 100%**. Num contexto de saúde do ativo, falhar a deteção de um erro (Falso Negativo) é catastrófico, logo, otimizar para 100% de Recall demonstra uma compreensão crítica do negócio.
* **Engineering Excellence:**
  * **Testes Estatísticos Pareados:** Em vez de assumir que o melhor modelo venceu por mero acaso, o pipeline aplica o teste t de Student (`ttest_rel`) sobre as amostras de validação cruzada para garantir significância estatística real. 

---

## 🛡️ AI Ethics & Best Practices

Num ambiente industrial, o "Black Box AI" gera desconfiança nos engenheiros de manutenção. Para garantir a **Interpretabilidade**:
* O pipeline incorpora **Permutation Feature Importance**, rankeando as características espectrais e temporais exatas que sinalizam a quebra.
* Inclui visualizações em 2D e 3D das fronteiras de decisão dos hiperplanos vetoriais.

---

## 🛣️ Future Work (Roadmap)

* **Model Serialization:** Guardar o pipeline do Scikit-Learn com `joblib` para servir inferências dinâmicas.
* **Edge Deployment:** Quantização do modelo SVM para implantação em microcontroladores acoplados à máquina (TinyML).

---
**Desenvolvido por Luis Tatin**


# ⚙️ Predictive-IoT: Industrial Fault Detection via Vibration Analysis

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![Scikit-Learn](https://img.shields.io/badge/Model-Scikit--Learn-F7931E?logo=scikit-learn)
![Scipy](https://img.shields.io/badge/Signal_Processing-Scipy-8CAAE6?logo=scipy)
![Pandas](https://img.shields.io/badge/Data_Engineering-Pandas-150458?logo=pandas)
![Industry 4.0](https://img.shields.io/badge/Domain-Industry_4.0-success)

This repository presents an advanced **Machine Learning solution for Predictive Maintenance**, focused on detecting anomalies in rotating machinery using triaxial vibration signals. The project covers the full lifecycle, from raw data ingestion (MATLAB files) to feature extraction in both time and frequency domains (Wavelets/FFT) and rigorous statistical evaluation.

---

## 📌 Project Overview (Business Vision)

* **Problem Statement:** On factory floors and in critical infrastructure, failures in mechanical components (such as bearings and motors) lead to unplanned downtime, safety risks, and high corrective maintenance costs.
* **Value Proposition:** By analyzing the vibrational signature of equipment in near real-time, this solution identifies microscopic degradation patterns before functional failure occurs. 
* **Impacted Success KPIs:**
  * Increase in **OEE** (Overall Equipment Effectiveness) by reducing downtime.
  * Maintenance cost optimization (reducing premature parts replacement).
  * Extension of **MTBF** (Mean Time Between Failures).

---

## 🏗️ Architecture & Engineering

The pipeline is heavily focused on Digital Signal Processing (DSP) and Statistics.

### 1. Data Pipeline (ETL)
* A dedicated script extracts raw sensor data from `.mat` formats (MATLAB time-series matrices), converts them into relational DataFrames, and labels the data as healthy (`0`) or faulty (`1`).

### 2. Signal Processing & Feature Engineering
* **Denoising:** Application of **Discrete Wavelet Transform (Daubechies 4)** to attenuate industrial noise without losing the transient peaks characteristic of cracks or friction.
* **Time Domain:** Calculation of robust metrics (RMS, Kurtosis, Skewness, Peak-to-Peak).
* **Frequency Domain:** Implementation of **Fast Fourier Transform (FFT)** to extract power spectra and median frequencies.
* **Feature Selection:** Removal of collinear variables (Pearson correlation > 0.90) to prevent overfitting.

### 3. Model Stack
* Various architectures were compared via `GridSearchCV` (K-NN, Decision Trees, LDA, SVMs with varied kernels).

---

## 🚀 Get Started (Quick Start)

**1. Generate the Structured Dataset**
The ETL script will read the `archive/Healthy` and `archive/Faulty` directories and aggregate everything into a single CSV.
```bash
python ETL.py
```

**2. Run the Training & Evaluation Pipeline**
This command will process the signals, extract features, evaluate the models, and generate performance plots.
```bash
python SVM_Modelo_e_Avaliacao.py
```

---

## 📊 Model Performance & Engineering Excellence

* **Top Performance:** The final classifier (Support Vector Machine with Linear Kernel) achieved an **Accuracy of 97.73%** and a **Recall/Sensitivity of 100%**. In an asset health context, missing a fault (False Negative) is catastrophic, so optimizing for 100% Recall demonstrates critical business acumen.
* **Engineering Excellence:**
  * **Paired Statistical Tests:** Instead of assuming the best model won by chance, the pipeline applies a Student's t-test (`ttest_rel`) on cross-validation folds to ensure true statistical significance. 

---

## 🛡️ AI Ethics & Best Practices

In an industrial environment, "Black Box AI" breeds distrust among maintenance engineers. To guarantee **Interpretability**:
* The pipeline incorporates **Permutation Feature Importance**, ranking the exact spectral and temporal features that signal a breakdown.
* It includes 2D and 3D visualizations of the vectorial hyperplane decision boundaries.

---

## 🛣️ Future Work (Roadmap)

* **Model Serialization:** Save the Scikit-Learn pipeline using `joblib` to serve dynamic inferences.
* **Edge Deployment:** Quantization of the SVM model for deployment on machine-attached microcontrollers (TinyML).

---
**Developed by Luis Tatin**
