# Detecção de Falhas em Máquinas Rotativas com Machine Learning (Pt-BR + EN)

Projeto completo de Machine Learning aplicado à **manutenção preditiva industrial**, incluindo **pipeline de ETL (extração, transformação e carga)** para conversão de dados brutos (`.mat`) em dados estruturados (`.csv`), seguido de modelagem e avaliação de algoritmos.

---

## Visão Geral

Em ambientes industriais, falhas em máquinas rotativas representam um dos principais fatores de:

- Paradas não planejadas
- Aumento de custos operacionais
- Redução da vida útil dos equipamentos

Este projeto propõe um pipeline completo para responder à seguinte pergunta:

> É possível detectar falhas potenciais em máquinas rotativas a partir de sinais de vibração?

---

## Arquitetura do Projeto

O projeto foi estruturado como um pipeline de dados e Machine Learning, dividido em três camadas principais:

### 🔹 1. ETL de Dados (Raw → Structured)

- Extração de dados no formato `.mat` (MATLAB)
- Conversão para `.csv` para uso em pipelines de ML
- Organização dos dados para análise e modelagem

Esse processo garante:

- Reprodutibilidade
- Interoperabilidade com ferramentas modernas (Python, Pandas, Scikit-learn)
- Padronização do dataset

---

### 🔹 2. Processamento de Sinais

#### Pré-processamento
- Normalização com **StandardScaler**
- Preservação da magnitude do sinal
- Redução de ruído

#### Transformação
- Aplicação de **Wavelet (db4)** para:
  - Filtrar ruídos
  - Destacar padrões transitórios de falha
  - Preparar o sinal para extração de características

---

### 🔹 3. Feature Engineering

#### Features Temporais
- Média
- Desvio padrão
- Curtose
- Assimetria
- RMS
- Pico-a-pico
- Energia

#### Features Espectrais
- Frequência média
- Desvio espectral
- Curtose espectral
- Potência de banda
- Mediana da frequência

---

### 🔹 4. Seleção de Features

- Análise de correlação
- Redução dimensional
- Seleção de um conjunto ótimo de variáveis (9 features)

---

### 🔹 5. Modelagem

Foram avaliados múltiplos algoritmos de classificação:

- SVM (Linear, Quadrático, Cúbico, Gaussiano)
- KNN (k=1, k=5, ponderado)
- Árvore de Decisão
- LDA (Linear Discriminant Analysis)

---

## 🏆 Resultados

O modelo com melhor desempenho foi o **SVM Linear**, apresentando:

- **Acurácia:** 97,73%
- **Sensibilidade (Recall):** 100%
- **Especificidade:** 95%

### 💡 Interpretação

- Detecção completa de falhas (sem falsos negativos)
- Baixa taxa de falsos positivos
- Alta confiabilidade para aplicações industriais críticas

---

## 📊 Diferenciais Técnicos

Este projeto demonstra competências avançadas em:

- ✔️ Engenharia de dados (ETL de `.mat` para `.csv`)
- ✔️ Processamento de sinais com Wavelets
- ✔️ Feature engineering (temporal e espectral)
- ✔️ Avaliação comparativa de múltiplos modelos
- ✔️ Construção de pipeline end-to-end de Machine Learning
- ✔️ Aplicação prática em contexto industrial (manutenção preditiva)

---

## Aplicações

- Manutenção preditiva
- Monitoramento de ativos industriais
- Indústria 4.0
- Sistemas de diagnóstico automatizado
- IoT industrial (IIoT)


- Roadmap
	•	Deploy do modelo como API (Flask / FastAPI)
	•	Integração com AWS (S3 + EC2 ou Lambda)
	•	Pipeline automatizado de ingestão de dados
	•	Monitoramento de modelo (MLOps)
	•	Dashboard interativo (Streamlit / Power BI)

👨‍💻 Autor

Luis Tatin
Engenheiro Mecatrônico | Data Science & Machine Learning | Instrutor de Tecnologia
  
---

# Fault Detection in Rotating Machinery using Machine Learning

A complete Machine Learning project applied to **industrial predictive maintenance**, including an **ETL pipeline (Extract, Transform, Load)** to convert raw data (`.mat`) into structured datasets (`.csv`), followed by model training and evaluation.

---

## Overview

In industrial environments, failures in rotating machinery are one of the main causes of:

- Unplanned downtime  
- Increased operational costs  
- Reduced equipment lifespan  

This project builds a full pipeline to answer the following question:

> Is it possible to detect potential faults in rotating machines using vibration signals?

---

## Project Architecture

The project is structured as a data and Machine Learning pipeline, divided into three main layers:

### 🔹 1. Data ETL (Raw → Structured)

- Extraction of data from `.mat` format (MATLAB)
- Conversion to `.csv` for Machine Learning workflows
- Data organization for analysis and modeling

This process ensures:

- Reproducibility  
- Interoperability with modern tools (Python, Pandas, Scikit-learn)  
- Dataset standardization  

---

### 🔹 2. Signal Processing

#### Preprocessing
- Normalization using **StandardScaler**
- Preservation of signal magnitude
- Noise reduction

#### Transformation
- Application of **Wavelet (db4)** to:
  - Filter noise  
  - Highlight transient fault patterns  
  - Prepare the signal for feature extraction  

---

### 🔹 3. Feature Engineering

#### Temporal Features
- Mean  
- Standard deviation  
- Kurtosis  
- Skewness  
- RMS  
- Peak-to-peak  
- Energy  

#### Spectral Features
- Mean frequency  
- Spectral standard deviation  
- Spectral kurtosis  
- Band power  
- Median frequency  

---

### 🔹 4. Feature Selection

- Correlation analysis  
- Dimensionality reduction  
- Selection of an optimal feature subset (9 features)  

---

### 🔹 5. Modeling

Multiple classification algorithms were evaluated:

- SVM (Linear, Quadratic, Cubic, Gaussian)  
- KNN (k=1, k=5, weighted)  
- Decision Tree  
- LDA (Linear Discriminant Analysis)  

---

## 🏆 Results

The best-performing model was the **Linear SVM**, achieving:

- **Accuracy:** 97.73%  
- **Sensitivity (Recall):** 100%  
- **Specificity:** 95%  

### 💡 Interpretation

- Complete detection of faults (no false negatives)  
- Low false positive rate  
- High reliability for critical industrial applications  

---

## 📊 Technical Highlights

This project demonstrates advanced skills in:

- ✔️ Data engineering (ETL from `.mat` to `.csv`)  
- ✔️ Signal processing using Wavelets  
- ✔️ Feature engineering (temporal and spectral domains)  
- ✔️ Comparative evaluation of multiple models  
- ✔️ End-to-end Machine Learning pipeline development  
- ✔️ Real-world industrial application (predictive maintenance)  

---

## Applications

- Predictive maintenance  
- Industrial asset monitoring  
- Industry 4.0  
- Automated fault diagnosis systems  
- Industrial IoT (IIoT)  

---

## Roadmap

- Deploy the model as an API (Flask / FastAPI)  
- Integrate with AWS (S3 + EC2 or Lambda)  
- Build an automated data ingestion pipeline  
- Implement model monitoring (MLOps)  
- Develop an interactive dashboard (Streamlit / Power BI)  

---

## 👨‍💻 Author

**Luis Tatin**  
Mechatronics Engineer | Data Science & Machine Learning | Technology Instructor  
