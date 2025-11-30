# 🔬 DataVision - Pipeline Automatizado de Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/streamlit-1.48+-red.svg)](https://streamlit.io/)
[![Scikit-learn](https://img.shields.io/badge/scikit--learn-1.7+-orange.svg)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

> Uma aplicação web interativa e completa para construção, treinamento, avaliação e deployment de modelos de Machine Learning, desenvolvida como projeto final do curso EBAC-SEMANTIX.

## 📋 Sobre o Projeto

DataVision é uma plataforma end-to-end de Machine Learning que automatiza todo o pipeline de ciência de dados, desde o carregamento e análise exploratória até o treinamento de modelos e predições. O projeto foi desenvolvido para democratizar o acesso a técnicas avançadas de ML, permitindo que usuários de diferentes níveis técnicos possam construir e avaliar modelos preditivos de forma visual e intuitiva.

## 🖼️ Screenshots

<div align="center">
  <table>
    <tr>
      <td align="center">
        <img src="https://i.imgur.com/fsbK0lx.png" width="200px" alt="Tela Inicial"><br>
        <sub><b>Tela Inicial</b></sub>
      </td>
      <td align="center">
        <img src="https://i.imgur.com/jMoNgjG.png" width="200px" alt="Análise de Target"><br>
        <sub><b>Análise de Target</b></sub>
      </td>
    </tr>
    <tr>
      <td align="center">
        <img src="https://i.imgur.com/njB3W3h.png" width="200px" alt="Modelagem"><br>
        <sub><b>Modelagem</b></sub>
      </td>
      <td align="center">
        <img src="https://i.imgur.com/jnfWeWV.png" width="200px" alt="Correlação de Dados"><br>
        <sub><b>Correlação de Dados</b></sub>
      </td>
    </tr>
  </table>
</div>

<details>
  <summary>🔍 Clique para ver imagens ampliadas</summary>
  <br>
  
  ### 🏠 Tela Inicial
  Interface principal com seleção de datasets e visão geral do projeto
  
  <img src="https://i.imgur.com/fsbK0lx.png" width="600px" alt="Tela Inicial - Ampliada">
  
  ---
  
  ### 🎯 Análise de Target
  Análise detalhada da variável alvo com distribuição de classes e estatísticas
  
  <img src="https://i.imgur.com/jMoNgjG.png" width="600px" alt="Análise de Target - Ampliada">
  
  ---
  
  ### 🤖 Modelagem
  Configuração e execução do pipeline de Machine Learning com métricas de performance
  
  <img src="https://i.imgur.com/njB3W3h.png" width="600px" alt="Modelagem - Ampliada">
  
  ---
  
  ### 📊 Correlação de Dados
  Matriz de correlação interativa para análise exploratória de dados
  
  <img src="https://i.imgur.com/jnfWeWV.png" width="600px" alt="Correlação de Dados - Ampliada">
</details>

### 🎯 Objetivos Principais

- **Automatização Completa**: Pipeline automatizado que realiza todo o fluxo de ML sem necessidade de código
- **Análise Exploratória Visual**: Visualizações interativas para entendimento profundo dos dados
- **Otimização Inteligente**: Seleção automática de features, balanceamento de classes e tuning de hiperparâmetros
- **Gerenciamento de Modelos**: Sistema completo para salvar, carregar e comparar diferentes modelos treinados
- **Predições em Produção**: Interface para realizar predições com modelos salvos em dados novos

## ✨ Funcionalidades Principais

### 📊 Pipeline de Machine Learning

- **Carregamento de Dados Flexível**:
  - Upload de arquivos CSV customizados
  - Datasets educacionais integrados (Iris, Wine, Breast Cancer)
  - Datasets de projetos reais (Credit Scoring, Hypertension, Phone Addiction)
  
- **Preprocessamento Automatizado**:
  - Tratamento inteligente de valores ausentes (mean, median, mode)
  - Detecção e remoção de outliers via IQR
  - Encoding automático de variáveis categóricas (Label Encoding, One-Hot Encoding)
  - Normalização de features (Standard, MinMax, Robust Scaler)
  - Limpeza de colunas problemáticas (datetime, alta cardinalidade)

- **Feature Engineering**:
  - Seleção automática de features (SelectKBest, RFE)
  - Extração de features temporais de colunas datetime
  - Análise de importância de features

- **Balanceamento de Classes**:
  - Implementação de SMOTE para oversampling
  - Configuração automática de class_weight
  - Detecção automática de datasets desbalanceados

- **Algoritmos de ML Suportados**:
  - Random Forest Classifier
  - Logistic Regression
  - Support Vector Machine (SVM)
  - XGBoost Classifier

- **Otimização de Hiperparâmetros**:
  - RandomizedSearchCV para busca eficiente
  - Grid de parâmetros pré-configurado para cada algoritmo
  - Validação cruzada estratificada (K-Fold)

- **Avaliação Robusta**:
  - Métricas completas: Accuracy, Precision, Recall, F1-Score
  - Validação cruzada com múltiplos folds
  - Matriz de confusão
  - Análise de performance por classe

### 📈 Visualizações Interativas

- **Análise Exploratória**:
  - Matriz de correlação com heatmap interativo
  - Distribuições univariadas e multivariadas
  - Scatter plots e pair plots
  - Box plots para análise por classe

- **Visualizações Específicas por Domínio**:
  - Análise por faixa etária (Credit Scoring)
  - Distribuição de pressão arterial (Hypertension)
  - Padrões de uso de smartphone (Phone Addiction)

- **Resultados do Pipeline**:
  - Gráficos de barras de métricas
  - Radar chart de performance
  - Evolução da accuracy ao longo do tempo
  - Comparação entre algoritmos

### 🤖 Gerenciamento de Modelos

- **Persistência**:
  - Salvamento automático de modelos treinados (.pkl)
  - Armazenamento de configurações e resultados (.json)
  - Controle de versões por timestamp

- **Predições**:
  - Upload de CSV para predições em lote
  - Entrada manual de dados para predições individuais
  - Predições em dados de exemplo para validação
  - Cálculo de probabilidades por classe

- **Análise Comparativa**:
  - Comparação de performance entre múltiplos modelos
  - Identificação do melhor modelo por métrica
  - Histórico completo de execuções
  - Análise de evolução temporal

### 🎛️ Configurações Avançadas

- Controle granular de todos os parâmetros do pipeline
- Configuração de algoritmos via interface
- Estratégias customizáveis de preprocessamento
- Validação cruzada configurável

## 🛠️ Tecnologias Utilizadas

### Core Stack
- **Python 3.9+** - Linguagem principal
- **Streamlit 1.48** - Framework web para interface
- **Pandas 2.3** - Manipulação e análise de dados
- **NumPy 2.0** - Computação numérica

### Machine Learning
- **Scikit-learn 1.7** - Algoritmos de ML, preprocessamento e avaliação
- **XGBoost** - Gradient boosting avançado
- **Imbalanced-learn** - Técnicas de balanceamento (SMOTE)

### Visualização
- **Plotly 6.3** - Gráficos interativos avançados
- **Matplotlib 3.9** - Visualizações estáticas
- **Seaborn 0.13** - Gráficos estatísticos

### Outras Bibliotecas
- **PyArrow 21.0** - Manipulação eficiente de dados
- **Joblib 1.5** - Serialização de modelos

## 📁 Estrutura do Projeto

```
DataVision/
│
├── app.py                          # Aplicação principal Streamlit
├── requirements.txt                # Dependências do projeto
├── README.md                       # Este arquivo
├── .gitignore                      # Arquivos ignorados pelo Git
│
├── src/                            # Código fonte principal
│   ├── __init__.py
│   ├── main.py                     # Renderização da página principal
│   └── sidebar.py                  # Renderização da sidebar
│
├── scripts/                        # Módulos do pipeline
│   ├── __init__.py
│   ├── data_loader.py              # Carregamento e validação de dados
│   │
│   ├── mainPage/                   # Componentes da página principal
│   │   ├── __init__.py
│   │   ├── pipeline.py             # Visualizações e análise exploratória
│   │   ├── model_run.py            # Resultados e análise de execuções
│   │   └── modelo.py               # Gerenciamento e predições com modelos
│   │
│   └── sidbar/                     # Componentes da sidebar
│       ├── __init__.py
│       ├── pipeline.py             # Configuração e execução do pipeline ML
│       └── modelo.py               # Sidebar para modelos salvos
│
├── data/                           # Diretório de dados
│   ├── raw/                        # Dados brutos
│   │   ├── credit_scoring.ftr
│   │   ├── hypertension_dataset.csv
│   │   └── teen_phone_addiction_dataset.csv
│   │
│   ├── models/                     # Modelos salvos (.pkl)
│   └── results/                    # Resultados das execuções (.json)
│
└── .streamlit/                     # Configurações do Streamlit
    └── secrets.toml                # Credenciais (não versionado)
```

## 🚀 Como Usar

### Instalação

1. **Clone o repositório**:
```bash
git clone https://github.com/Dieghonm/DataVision.git
cd DataVision
```

2. **Crie um ambiente virtual**:
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

3. **Instale as dependências**:
```bash
pip install -r requirements.txt
```

4. **Execute a aplicação**:
```bash
streamlit run app.py
```

5. **Acesse no navegador**: `http://localhost:8501`

### Fluxo de Uso

#### 1️⃣ Pipeline de Dados

1. **Selecione um Dataset**:
   - Faça upload de um CSV customizado
   - Ou escolha um dos datasets pré-carregados

2. **Analise os Dados**:
   - Visualize estatísticas descritivas
   - Explore correlações e distribuições
   - Identifique a variável target

3. **Configure o Pipeline**:
   - Escolha o algoritmo de ML
   - Configure preprocessamento
   - Ative otimizações (feature selection, balanceamento, tuning)

4. **Execute o Pipeline**:
   - Clique em "Executar Pipeline"
   - Acompanhe o progresso de cada etapa
   - Analise os resultados e métricas

#### 2️⃣ Utilizar Modelo Salvo

1. **Carregue um Modelo**:
   - Selecione um modelo da lista
   - Visualize informações e performance

2. **Faça Predições**:
   - Upload de CSV para predições em lote
   - Entrada manual para casos individuais
   - Use dados de exemplo para testes

3. **Análise Comparativa**:
   - Compare múltiplos modelos
   - Identifique o melhor modelo
   - Visualize evolução temporal

## 📊 Datasets Incluídos

### Datasets Educacionais (Scikit-learn)

**Iris Dataset**
- Problema: Classificação de espécies de flores
- Classes: 3 (Setosa, Versicolor, Virginica)
- Features: 4 medidas morfológicas
- Amostras: 150

**Wine Dataset**
- Problema: Classificação de vinhos por origem
- Classes: 3 cultivares diferentes
- Features: 13 análises químicas
- Amostras: 178

**Breast Cancer**
- Problema: Diagnóstico de câncer de mama
- Classes: 2 (Maligno, Benigno)
- Features: 30 características celulares
- Amostras: 569

### Datasets de Projetos Reais

**Credit Scoring**
- Problema: Análise de risco de crédito
- Objetivo: Prever aprovação de empréstimo
- Aplicação: Bancos e fintechs

**Hypertension**
- Problema: Predição de hipertensão arterial
- Objetivo: Identificar pacientes de risco
- Aplicação: Diagnóstico preventivo

**Phone Addiction**
- Problema: Identificação de vício em smartphones
- Objetivo: Detectar uso problemático
- Aplicação: Saúde mental e bem-estar digital

## 🎓 Conceitos de ML Implementados

### Preprocessamento
- Tratamento de missing values
- Detecção e remoção de outliers
- Normalização e padronização
- Encoding de variáveis categóricas
- Feature scaling

### Feature Engineering
- Seleção automática de features (SelectKBest, RFE)
- Extração de features temporais
- Análise de importância

### Tratamento de Desbalanceamento
- SMOTE (Synthetic Minority Over-sampling)
- Class weight balancing
- Detecção automática de desbalanceamento

### Otimização
- Hyperparameter tuning com RandomizedSearchCV
- Cross-validation estratificada
- Grid search de parâmetros

### Avaliação
- Métricas de classificação completas
- Matriz de confusão
- Validação cruzada K-Fold
- Análise de performance por classe

## 🔧 Configurações Disponíveis

### Algoritmos
- Random Forest (n_estimators, max_depth, min_samples_split, min_samples_leaf)
- Logistic Regression (C, solver, max_iter)
- SVM (C, kernel, gamma)
- XGBoost (n_estimators, max_depth, learning_rate)

### Preprocessamento
- **Normalização**: Standard, MinMax, Robust, None
- **Encoding**: Label Encoding, One-Hot Encoding
- **Missing Values**: Mean, Median, Mode
- **Outliers**: Remoção via IQR

### Otimizações
- Feature Selection (SelectKBest, RFE)
- Class Balancing (SMOTE, Class Weight)
- Hyperparameter Tuning (RandomizedSearchCV)
- Cross-Validation (3-10 folds)

## 📈 Métricas e Avaliação

O sistema calcula automaticamente:

- **Accuracy**: Proporção de predições corretas
- **Precision**: Relevância das predições positivas
- **Recall**: Cobertura dos casos positivos
- **F1-Score**: Média harmônica entre precision e recall
- **Cross-Validation Score**: Validação com múltiplos folds
- **Confusion Matrix**: Matriz de confusão para análise detalhada

## 🎨 Interface e Usabilidade

### Características da Interface

- **Design Intuitivo**: Interface limpa e organizada
- **Visualizações Interativas**: Gráficos Plotly totalmente interativos
- **Feedback em Tempo Real**: Progress bars e status de cada etapa
- **Responsiva**: Adaptável a diferentes tamanhos de tela
- **Documentação Inline**: Tooltips e help text em cada configuração

### Componentes Principais

1. **Sidebar**: Configuração do pipeline e navegação
2. **Área Principal**: Visualizações e resultados
3. **Tabs**: Organização de diferentes análises
4. **Expandables**: Informações detalhadas sob demanda
5. **Métricas Cards**: Visualização rápida de KPIs

## 🔐 Segurança e Boas Práticas

- Validação robusta de dados de entrada
- Tratamento de erros em todas as etapas
- Limpeza automática de dados problemáticos
- Logs informativos de cada operação
- Arquivos sensíveis no .gitignore
- Ambiente virtual isolado

## 🚧 Melhorias Futuras

- [ ] Suporte para regressão
- [ ] Mais algoritmos (Neural Networks, Ensemble Methods)
- [ ] Export de relatórios em PDF
- [ ] API REST para predições
- [ ] Suporte para séries temporais
- [ ] Deploy automatizado
- [ ] Integração com MLflow
- [ ] Suporte para deep learning

## 👨‍💻 Autor

**Diegho Moraes**
- GitHub: [@Dieghonm](https://github.com/Dieghonm)
- LinkedIn: [Diegho Moraes](https://linkedin.com/in/diegho-moraes)
- Localização: Rio de Janeiro, RJ, Brasil

## 📝 Licença

Este projeto está sob a licença MIT. Veja o arquivo [LICENSE](LICENSE) para mais detalhes.

## 🙏 Agradecimentos

- **EBAC e SEMANTIX** pelo curso e conhecimento compartilhado
- **Comunidade Streamlit** pela excelente framework
- **Scikit-learn** pelas ferramentas de ML
- **Plotly** pelas visualizações interativas

---

⭐ Se este projeto foi útil para você, considere dar uma estrela no repositório!

**Desenvolvido com ❤️ para democratizar o acesso ao Machine Learning**
