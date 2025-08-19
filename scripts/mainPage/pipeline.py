import streamlit as st
import pandas as pd
from scripts.data_loader import load_dataset

def pipeline_main():
    """Página principal do pipeline de dados"""
    
    # Verificar se pipeline_config existe no session_state
    if "pipeline_config" not in st.session_state or st.session_state.pipeline_config is None:
        _render_welcome()
        return
    
    # Obter configurações do pipeline
    pipeline_config = st.session_state.pipeline_config
    data_source = pipeline_config.get("data_source")
    uploaded_file = pipeline_config.get("uploaded_file")
    
    # Se não há fonte de dados ou é upload sem arquivo, mostrar welcome
    if not data_source or (data_source == "upload" and not uploaded_file):
        _render_welcome()
        return
    
    # Carregar o dataset
    try:
        df, nome_dataset = load_dataset(data_source, uploaded_file)
        
        if df is not None:
            # Armazenar o DataFrame no session_state para uso posterior
            st.session_state.DF = df
            
            # Mostrar informações do dataset
            st.write(f"### 📊 Dataset: {nome_dataset}")
            
            # Criar tabs para melhor organização
            tab1, tab2, tab3 = st.tabs(["👁️ Prévia dos Dados", "📈 Estatísticas", "🔍 Informações Gerais"])
            
            with tab1:
                st.write("**Primeiras 10 linhas:**")
                st.dataframe(df.head(10), use_container_width=True)
            
            with tab2:
                st.write("**Estatísticas descritivas:**")
                # Separar colunas numéricas e categóricas
                numeric_cols = df.select_dtypes(include=['number']).columns
                categorical_cols = df.select_dtypes(include=['object', 'category']).columns
                
                if len(numeric_cols) > 0:
                    st.write("*Colunas numéricas:*")
                    st.dataframe(df[numeric_cols].describe(), use_container_width=True)
                
                if len(categorical_cols) > 0:
                    st.write("*Colunas categóricas:*")
                    st.dataframe(df[categorical_cols].describe(), use_container_width=True)
            
            with tab3:
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("📏 Linhas", df.shape[0])
                    st.metric("📊 Colunas", df.shape[1])
                    st.metric("💾 Tamanho em Memória", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                with col2:
                    # Calcular valores faltantes
                    missing_count = df.isnull().sum().sum()
                    missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
                    
                    st.metric("❓ Valores Faltantes", f"{missing_count} ({missing_percent:.1f}%)")
                    st.metric("🔢 Colunas Numéricas", len(numeric_cols))
                    st.metric("📝 Colunas Categóricas", len(categorical_cols))
                
                # Informações sobre tipos de dados
                st.write("**Tipos de Dados:**")
                dtype_info = df.dtypes.value_counts()
                for dtype, count in dtype_info.items():
                    st.write(f"- {dtype}: {count} colunas")
                
                # Se há valores faltantes, mostrar detalhes
                if missing_count > 0:
                    st.write("**Valores Faltantes por Coluna:**")
                    missing_data = df.isnull().sum()
                    missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
                    
                    for col, missing in missing_data.items():
                        percent = (missing / len(df)) * 100
                        st.write(f"- **{col}**: {missing} ({percent:.1f}%)")
            
            # Adicionar visualizações específicas para cada dataset
            _render_dataset_visualizations(df, data_source, nome_dataset)
            
            # Mostrar informações sobre o target se existir
            if 'target' in df.columns:
                st.markdown("---")
                st.write("### 🎯 Informações do Target")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Distribuição das Classes:**")
                    target_counts = df['target'].value_counts()
                    
                    # Se existe target_name, usar ela para mostrar os nomes
                    if 'target_name' in df.columns:
                        target_name_counts = df['target_name'].value_counts()
                        st.write(target_name_counts)
                    else:
                        st.write(target_counts)
                
                with col2:
                    # Gráfico de barras simples da distribuição
                    if 'target_name' in df.columns:
                        st.bar_chart(df['target_name'].value_counts())
                    else:
                        st.bar_chart(df['target'].value_counts())
        else:
            st.error("❌ Erro ao carregar o dataset. Verifique o arquivo ou tente outro dataset.")
            _render_welcome()
            
    except Exception as e:
        st.error(f"❌ Erro ao processar o dataset: {str(e)}")
        _render_welcome()


def _render_dataset_visualizations(df, data_source, nome_dataset):
    """Renderiza visualizações específicas para cada dataset"""
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8')
    
    st.markdown("---")
    st.subheader("📈 Visualizações do Dataset")
    
    # Verificar se tem target para visualizações específicas
    has_target = 'target' in df.columns or 'target_name' in df.columns
    
    # Verificar se há uma seleção manual de target salva
    manual_target_selected = False
    if 'manual_target' in st.session_state and data_source in st.session_state.manual_target:
        manual_target_col = st.session_state.manual_target[data_source]
        if manual_target_col in df.columns:
            has_target = True
            manual_target_selected = True
    
    # Colunas numéricas para análise
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    # Criar tabs para diferentes tipos de visualização
    viz_tabs = st.tabs(["🔍 Correlações", "📊 Distribuições", "🎯 Target Analysis", "📈 Específicas do Dataset"])
    
    with viz_tabs[0]:  # Correlações
        if len(numeric_cols) > 1:
            st.write("**Matriz de Correlação:**")
            
            # Calcular correlação
            corr_matrix = df[numeric_cols].corr()
            
            # Plotly heatmap
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Matriz de Correlação",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há colunas numéricas suficientes para matriz de correlação")
    
    with viz_tabs[1]:  # Distribuições
        if len(numeric_cols) > 0:
            st.write("**Distribuições das Variáveis Numéricas:**")
            
            # Selecionar colunas para plotar
            cols_to_plot = st.multiselect(
                "Selecione as colunas para visualizar:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if cols_to_plot:
                # Criar subplots
                n_cols = min(2, len(cols_to_plot))
                n_rows = (len(cols_to_plot) + n_cols - 1) // n_cols
                
                fig = make_subplots(
                    rows=n_rows, cols=n_cols,
                    subplot_titles=cols_to_plot,
                    vertical_spacing=0.08
                )
                
                for i, col in enumerate(cols_to_plot):
                    row = i // n_cols + 1
                    col_pos = i % n_cols + 1
                    
                    fig.add_trace(
                        go.Histogram(x=df[col], name=col, showlegend=False),
                        row=row, col=col_pos
                    )
                
                fig.update_layout(height=300 * n_rows, title_text="Distribuições")
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há colunas numéricas para mostrar distribuições")
    
    with viz_tabs[2]:  # Target Analysis
        if has_target:
            st.write("**Análise da Variável Target:**")
            
            # Determinar qual coluna usar como target
            if manual_target_selected:
                target_col = st.session_state.manual_target[data_source]
                col_info, col_reset = st.columns([3, 1])
                with col_info:
                    st.info(f"🎯 Usando '{target_col}' como variável target (seleção manual)")
                with col_reset:
                    if st.button("🔄 Alterar", help="Clique para selecionar outra coluna target"):
                        del st.session_state.manual_target[data_source]
                        st.rerun()
            else:
                target_col = 'target_name' if 'target_name' in df.columns else 'target'
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Distribuição do target
                target_counts = df[target_col].value_counts()
                fig = px.pie(
                    values=target_counts.values, 
                    names=target_counts.index,
                    title="Distribuição das Classes"
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                # Bar chart
                fig = px.bar(
                    x=target_counts.index, 
                    y=target_counts.values,
                    title="Contagem por Classe",
                    labels={'x': 'Classe', 'y': 'Quantidade'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Box plots das variáveis numéricas por target
            if len(numeric_cols) > 0:
                st.write("**Distribuição das Variáveis por Classe:**")
                
                selected_var = st.selectbox(
                    "Selecione uma variável para análise:",
                    numeric_cols,
                    key="target_analysis"
                )
                
                fig = px.box(
                    df, 
                    x=target_col, 
                    y=selected_var,
                    title=f"Distribuição de {selected_var} por Classe"
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            # Permitir seleção manual da coluna target
            st.write("**🎯 Selecione a Variável Target:**")
            
            # Filtrar colunas categóricas e numéricas que podem ser target
            possible_targets = []
            for col in df.columns:
                unique_values = df[col].nunique()
                # Considerar colunas com poucos valores únicos (2-10) como possíveis targets
                if 2 <= unique_values <= 10:
                    possible_targets.append(col)
            
            if possible_targets:
                selected_target = st.selectbox(
                    "Escolha a coluna que representa a variável target:",
                    ["Nenhuma"] + possible_targets,
                    help="Selecione a coluna que contém a variável que você quer prever"
                )
                
                if selected_target != "Nenhuma":
                    # Criar uma versão temporária do DataFrame com o target selecionado
                    df_temp = df.copy()
                    df_temp['selected_target'] = df_temp[selected_target]
                    
                    # Análises com o target selecionado
                    st.write(f"**Análise da Variável: {selected_target}**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribuição do target selecionado
                        target_counts = df_temp['selected_target'].value_counts()
                        fig = px.pie(
                            values=target_counts.values, 
                            names=target_counts.index,
                            title=f"Distribuição de {selected_target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Bar chart
                        fig = px.bar(
                            x=target_counts.index, 
                            y=target_counts.values,
                            title=f"Contagem por {selected_target}",
                            labels={'x': selected_target, 'y': 'Quantidade'}
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Mostrar estatísticas
                    st.write("**Estatísticas da Variável Target:**")
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.metric("Classes Únicas", df_temp['selected_target'].nunique())
                    with col2:
                        st.metric("Valores Faltantes", df_temp['selected_target'].isnull().sum())
                    with col3:
                        most_common = target_counts.index[0]
                        percentage = (target_counts.iloc[0] / len(df_temp)) * 100
                        st.metric("Classe Majoritária", f"{most_common} ({percentage:.1f}%)")
                    
                    # Box plots das variáveis numéricas por target
                    if len(numeric_cols) > 0:
                        st.write("**Distribuição das Variáveis por Classe:**")
                        
                        selected_var = st.selectbox(
                            "Selecione uma variável para análise:",
                            numeric_cols,
                            key="manual_target_analysis"
                        )
                        
                        fig = px.box(
                            df_temp, 
                            x='selected_target', 
                            y=selected_var,
                            title=f"Distribuição de {selected_var} por {selected_target}"
                        )
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Correlação com variáveis numéricas
                    if len(numeric_cols) > 0 and df_temp['selected_target'].dtype in ['int64', 'float64']:
                        st.write("**Correlação com Variáveis Numéricas:**")
                        
                        # Calcular correlações
                        correlations = []
                        for col in numeric_cols:
                            corr = df_temp[col].corr(df_temp['selected_target'])
                            if not pd.isna(corr):
                                correlations.append({'Variable': col, 'Correlation': corr})
                        
                        if correlations:
                            corr_df = pd.DataFrame(correlations)
                            corr_df = corr_df.sort_values('Correlation', key=abs, ascending=False)
                            
                            fig = px.bar(
                                corr_df, 
                                x='Correlation', 
                                y='Variable',
                                orientation='h',
                                title=f"Correlação das Variáveis com {selected_target}",
                                color='Correlation',
                                color_continuous_scale='RdBu'
                            )
                            st.plotly_chart(fig, use_container_width=True)
                    
                    # Adicionar ao session_state para uso posterior
                    if 'manual_target' not in st.session_state:
                        st.session_state.manual_target = {}
                    st.session_state.manual_target[data_source] = selected_target
                    
                    st.success(f"✅ Variável target '{selected_target}' selecionada! Esta informação será lembrada para este dataset.")
                
                else:
                    st.info("👆 Selecione uma coluna target para ver as análises")
            else:
                st.warning("⚠️ Não foram encontradas colunas adequadas para serem target (colunas com 2-10 valores únicos)")
                st.info("💡 Dica: Variáveis target geralmente têm poucos valores únicos (ex: classes, categorias)")
    
    with viz_tabs[3]:  # Específicas do Dataset
        _render_specific_visualizations(df, data_source, numeric_cols)


def _render_specific_visualizations(df, data_source, numeric_cols):
    """Renderiza visualizações específicas para cada dataset"""
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    if data_source == "Credit":
        st.write("**📊 Análises Específicas - Credit Scoring:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = px.histogram(df, x='age', title="Distribuição de Idade", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'income' in df.columns:
                fig = px.histogram(df, x='income', title="Distribuição de Renda", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        
        # Análise de risco por idade
        if 'age' in df.columns and 'target_name' in df.columns:
            # Criar faixas etárias
            df_viz = df.copy()
            df_viz['age_group'] = pd.cut(df['age'], bins=[0, 25, 35, 45, 55, 100], 
                                       labels=['18-25', '26-35', '36-45', '46-55', '55+'])
            
            fig = px.histogram(df_viz, x='age_group', color='target_name', 
                             title="Distribuição de Aprovação por Faixa Etária",
                             barmode='group')
            st.plotly_chart(fig, use_container_width=True)
    
    elif data_source == "Hipertension":
        st.write("**🩺 Análises Específicas - Hipertensão:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'age' in df.columns:
                fig = px.histogram(df, x='age', title="Distribuição de Idade", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'bmi' in df.columns:
                fig = px.histogram(df, x='bmi', title="Distribuição de IMC", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        # Pressão arterial por target
        if 'systolic_bp' in df.columns and 'diastolic_bp' in df.columns and 'target_name' in df.columns:
            fig = px.scatter(df, x='systolic_bp', y='diastolic_bp', color='target_name',
                           title="Pressão Sistólica vs Diastólica por Diagnóstico")
            st.plotly_chart(fig, use_container_width=True)
        
        # Fatores de risco
        risk_factors = ['smoking', 'alcohol', 'exercise', 'family_history']
        available_factors = [f for f in risk_factors if f in df.columns]
        
        if available_factors:
            st.write("**Fatores de Risco:**")
            factor_data = []
            
            for factor in available_factors:
                if df[factor].dtype in ['int64', 'float64']:
                    positive_rate = (df[factor] == 1).mean() * 100
                    factor_data.append({'Fator': factor.replace('_', ' ').title(), 'Taxa (%)': positive_rate})
            
            if factor_data:
                import pandas as pd
                factor_df = pd.DataFrame(factor_data)
                fig = px.bar(factor_df, x='Fator', y='Taxa (%)', 
                           title="Taxa de Fatores de Risco na População")
                st.plotly_chart(fig, use_container_width=True)
    
    elif data_source == "Phone addiction":
        st.write("**📱 Análises Específicas - Vício em Smartphone:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'daily_usage_hours' in df.columns:
                fig = px.histogram(df, x='daily_usage_hours', title="Horas de Uso Diário", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'sleep_hours' in df.columns:
                fig = px.histogram(df, x='sleep_hours', title="Horas de Sono", nbins=10)
                st.plotly_chart(fig, use_container_width=True)
        
        # Relação entre uso e sono
        if 'daily_usage_hours' in df.columns and 'sleep_hours' in df.columns:
            color_col = 'target_name' if 'target_name' in df.columns else None
            fig = px.scatter(df, x='daily_usage_hours', y='sleep_hours', color=color_col,
                           title="Relação entre Uso Diário e Horas de Sono")
            st.plotly_chart(fig, use_container_width=True)
        
        # Análise por idade
        if 'age' in df.columns and 'daily_usage_hours' in df.columns:
            fig = px.box(df, x='age', y='daily_usage_hours', 
                        title="Uso Diário por Idade")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Para datasets clássicos (iris, wine, breast_cancer)
        st.write(f"**📊 Análises Específicas - {data_source.title()}:**")
        
        if len(numeric_cols) >= 2:
            # Scatter plot das duas primeiras variáveis
            color_col = 'target_name' if 'target_name' in df.columns else 'target' if 'target' in df.columns else None
            
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=color_col,
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Pairplot para datasets pequenos
        if len(df) < 1000 and len(numeric_cols) <= 6:
            st.write("**Matriz de Scatter Plots:**")
            
            # Selecionar até 4 variáveis para o pairplot
            plot_vars = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
            
            if len(plot_vars) >= 2:
                fig = px.scatter_matrix(df, dimensions=plot_vars, 
                                      color='target_name' if 'target_name' in df.columns else None,
                                      title="Matriz de Correlações")
                fig.update_layout(height=600)
                st.plotly_chart(fig, use_container_width=True)


def _render_welcome():
    """Renderiza tela de boas-vindas"""
    st.markdown("""
    # 🚀 DataVision EBAC SEMANTIX
    
    Esta aplicação permite que você configure e execute pipelines de Machine Learning 
    de forma visual e interativa.
    
    ### Como usar:
    1. **📂 Selecione um dataset** na barra lateral
    2. **👁️ Visualize os dados** antes do processamento
    3. **🤖 Configure o algoritmo** e seus parâmetros
    4. **📊 Defina as métricas** de avaliação
    5. **🚀 Execute o pipeline** e veja os resultados
    
    **👈 Comece selecionando um dataset na barra lateral!**
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
    
    # Histórico de execuções (se existir)
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