import streamlit as st
from zold.data_loader import load_dataset

def pipeline_page(params):
    """Página principal do pipeline de dados"""
    data_source = params.get("data_source")
    uploaded_file = params.get("uploaded_file")
    etapas = params.get("etapas", [])

    df, nome_dataset = load_dataset(data_source, uploaded_file)

    if df is not None:
        st.write(f"### {nome_dataset}")
        st.dataframe(df.head())
        st.write(f"Shape: {df.shape[0]} linhas × {df.shape[1]} colunas")
        st.write("### Estatísticas descritivas:")
        st.write(df.describe())

        if etapas:
            st.success(f"Etapas selecionadas: {', '.join(etapas)}")
        else:
            st.info("Nenhuma etapa do pipeline foi selecionada.")
    else:
        _render_welcome()


def _render_welcome():
    """Renderiza tela de boas-vindas"""
    st.markdown("""
    Esta aplicação permite que você configure e execute pipelines de Machine Learning 
    de forma visual e interativa.
    
    ### Como usar:
    1. **Selecione um dataset** na barra lateral
    2. **Visualize os dados** antes do processamento
    3. **Configure o algoritmo** e seus parâmetros
    4. **Defina as métricas** de avaliação
    5. **Execute o pipeline** e veja os resultados
    
    **Comece selecionando um dataset na barra lateral!**
    """)
    
    # Seção de Datasets Disponíveis
    st.markdown("---")
    st.subheader("📊 Datasets Disponíveis")
    st.markdown("Conheça os datasets que você pode usar nesta aplicação:")
    
    # Datasets Clássicos (Sklearn)
    st.markdown("### 📚 Datasets Clássicos (Educacionais)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("🌸 Iris Dataset"):
            st.markdown("""
            **Iris Dataset**
            - **Problema**: Classificação de espécies de flores íris
            - **Classes**: 3 (Setosa, Versicolor, Virginica)  
            - **Features**: 4 (comprimento/largura de pétalas e sépalas)
            - **Amostras**: 150 (50 por classe)
            - **Uso**: Perfeito para iniciantes - classificação multiclasse simples
            - **Origem**: Ronald Fisher (1936)
            """)
    
    with col2:
        with st.expander("🍷 Wine Dataset"):
            st.markdown("""
            **Wine Dataset**
            - **Problema**: Classificação de vinhos por origem
            - **Classes**: 3 (diferentes cultivares)
            - **Features**: 13 (análises químicas: álcool, ácido málico, etc.)
            - **Amostras**: 178 vinhos
            - **Uso**: Classificação com mais complexidade
            - **Origem**: Vinhos da região de Piemonte, Itália
            """)
    
    with col3:
        with st.expander("🎗️ Breast Cancer"):
            st.markdown("""
            **Breast Cancer Dataset**
            - **Problema**: Diagnóstico de câncer de mama
            - **Classes**: 2 (Maligno, Benigno)
            - **Features**: 30 (características dos núcleos celulares)
            - **Amostras**: 569 casos
            - **Uso**: Classificação binária - aplicação médica importante
            - **Origem**: Hospital da Universidade de Wisconsin
            """)
    
    # Datasets Personalizados
    st.markdown("### 🎯 Datasets Personalizados (Projetos Reais)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("💳 Credit Scoring"):
            st.markdown("""
            **Credit Scoring Dataset**
            - **Problema**: Análise de risco de crédito
            - **Objetivo**: Prever se um cliente vai pagar o empréstimo
            - **Tipo**: Classificação binária (Aprovado/Negado)
            - **Aplicação**: Bancos e fintechs
            - **Importância**: Decisões financeiras automatizadas
            - **Desafios**: Balanceamento, interpretabilidade
            """)
    
    with col2:
        with st.expander("🩺 Hypertension"):
            st.markdown("""
            **Hypertension Dataset**
            - **Problema**: Predição de hipertensão arterial
            - **Objetivo**: Identificar pacientes com risco de hipertensão
            - **Tipo**: Classificação médica
            - **Aplicação**: Diagnóstico preventivo
            - **Importância**: Saúde pública - prevenção de doenças cardiovasculares
            - **Features**: Dados demográficos, estilo de vida, exames
            """)
    
    with col3:
        with st.expander("📱 Phone Addiction"):
            st.markdown("""
            **Teen Phone Addiction Dataset**
            - **Problema**: Identificação de vício em smartphones
            - **Objetivo**: Detectar adolescentes com uso problemático do celular
            - **Tipo**: Classificação comportamental
            - **Aplicação**: Saúde mental, bem-estar digital
            - **Importância**: Problema crescente na era digital
            - **Features**: Padrões de uso, comportamento, dados psicológicos
            """)
    
    # Recursos da aplicação
    st.markdown("---")
    st.markdown("### ⚡ Recursos da Aplicação")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **📊 Análise de Dados:**
        - Prévia interativa dos datasets
        - Estatísticas descritivas automáticas
        - Visualizações exploratórias
        - Análise de qualidade dos dados
        - Detecção de valores faltantes
        - Análise de correlações
        """)
    
    with feature_col2:
        st.markdown("""
        **🤖 Machine Learning:**
        - Múltiplos algoritmos (RF, SVM, LogReg)
        - Configuração de hiperparâmetros
        - Cross-validation automática
        - Métricas de avaliação completas
        - Visualizações de resultados
        - Histórico de experimentos
        """)
    
    # Histórico de execuções
    if 'executions' in st.session_state and st.session_state.executions:
        st.markdown("---")
        st.subheader("📋 Histórico de Execuções")
        
        for i, execution in enumerate(reversed(st.session_state.executions[-5:])):
            with st.expander(f"Execução {len(st.session_state.executions) - i} - {execution['timestamp'].strftime('%H:%M:%S')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Algoritmo:**", execution['config']['model']['algorithm'])
                    st.write("**Status:**", execution['results'].get('status', 'unknown'))
                with col2:
                    if 'evaluation' in execution['results']:
                        metrics = execution['results']['evaluation']['metrics']
                        for metric, value in metrics.items():
                            if isinstance(value, (int, float)):
                                st.metric(metric.title(), f"{value:.4f}")