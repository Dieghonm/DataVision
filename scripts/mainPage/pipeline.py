import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scripts.data_loader import load_dataset

class PipelineVisualizer:
    def __init__(self):
        self.viz_config = {
            'height': 500,
            'template': 'plotly_white'
        }

    def render_data_overview(self, df, nome_dataset):
        st.write(f"### Dataset: {nome_dataset}")
        
        tab1, tab2, tab3 = st.tabs(["Prévia dos Dados", "Estatísticas", "Informações Gerais"])
        
        with tab1:
            st.write("**Primeiras 5 linhas:**")
            st.dataframe(df.head(5), use_container_width=True)
        
        with tab2:
            self._render_statistics(df)
        
        with tab3:
            self._render_general_info(df)

    def _render_statistics(self, df):
        st.write("**Estatísticas descritivas:**")
        numeric_cols = df.select_dtypes(include=['number']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numeric_cols) > 0:
            st.write("*Colunas numéricas:*")
            st.dataframe(df[numeric_cols].describe(), use_container_width=True)
        
        if len(categorical_cols) > 0:
            st.write("*Colunas categóricas:*")
            st.dataframe(df[categorical_cols].describe(), use_container_width=True)

    def _render_general_info(self, df):
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Linhas", df.shape[0])
            st.metric("Colunas", df.shape[1])
            memory_usage = df.memory_usage(deep=True).sum() / 1024
            st.metric("Tamanho em Memória", f"{memory_usage:.1f} KB")
        
        with col2:
            missing_count = df.isnull().sum().sum()
            missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
            numeric_cols = df.select_dtypes(include=['number']).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            st.metric("Valores Faltantes", f"{missing_count} ({missing_percent:.1f}%)")
            st.metric("Colunas Numéricas", len(numeric_cols))
            st.metric("Colunas Categóricas", len(categorical_cols))
        
        self._render_column_info(df, missing_count)

    def _render_column_info(self, df, missing_count):
        st.write("**Tipos de Dados:**")
        dtype_info = df.dtypes.value_counts()
        for dtype, count in dtype_info.items():
            st.write(f"- {dtype}: {count} colunas")
        
        if missing_count > 0:
            st.write("**Valores Faltantes por Coluna:**")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0].sort_values(ascending=False)
            
            for col, missing in missing_data.items():
                percent = (missing / len(df)) * 100
                st.write(f"- **{col}**: {missing} ({percent:.1f}%)")

    def render_visualizations(self, df, data_source, nome_dataset):
        st.markdown("---")
        st.subheader("Visualizações do Dataset")
        
        has_target = 'target' in df.columns or 'target_name' in df.columns
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        viz_tabs = st.tabs(["Correlações", "Distribuições", "Target Analysis", "Específicas"])
        
        with viz_tabs[0]:
            self._render_correlation_matrix(df, numeric_cols)
        
        with viz_tabs[1]:
            self._render_distributions(df, numeric_cols)
        
        with viz_tabs[2]:
            self._render_target_analysis(df, data_source, has_target, numeric_cols)
        
        with viz_tabs[3]:
            self._render_specific_visualizations(df, data_source, numeric_cols)

    def _render_correlation_matrix(self, df, numeric_cols):
        if len(numeric_cols) > 1:
            st.write("**Matriz de Correlação:**")
            
            corr_matrix = df[numeric_cols].corr()
            
            fig = px.imshow(
                corr_matrix, 
                text_auto=True, 
                aspect="auto",
                title="Matriz de Correlação",
                color_continuous_scale='RdBu'
            )
            fig.update_layout(height=self.viz_config['height'])
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há colunas numéricas suficientes para matriz de correlação")

    def _render_distributions(self, df, numeric_cols):
        if len(numeric_cols) > 0:
            st.write("**Distribuições das Variáveis Numéricas:**")
            
            cols_to_plot = st.multiselect(
                "Selecione as colunas para visualizar:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if cols_to_plot:
                self._create_distribution_plots(df, cols_to_plot)
        else:
            st.info("Não há colunas numéricas para mostrar distribuições")

    def _create_distribution_plots(self, df, cols_to_plot):
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

    def _render_target_analysis(self, df, data_source, has_target, numeric_cols):
        if has_target:
            target_col = 'target_name' if 'target_name' in df.columns else 'target'
            self._render_target_visualizations(df, target_col, numeric_cols)
        else:
            self._render_target_selection(df, data_source, numeric_cols)

    def _render_target_visualizations(self, df, target_col, numeric_cols):
        st.write("**Análise da Variável Target:**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = df[target_col].value_counts()
            fig = px.pie(
                values=target_counts.values, 
                names=target_counts.index,
                title="Distribuição das Classes"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=target_counts.index, 
                y=target_counts.values,
                title="Contagem por Classe",
                labels={'x': 'Classe', 'y': 'Quantidade'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        if len(numeric_cols) > 0:
            self._render_target_vs_features(df, target_col, numeric_cols)

    def _render_target_vs_features(self, df, target_col, numeric_cols):
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

    def _render_target_selection(self, df, data_source, numeric_cols):
        st.write("**Selecione a Variável Target:**")
        
        possible_targets = [col for col in df.columns if 2 <= df[col].nunique() <= 10]
        
        if possible_targets:
            last_column = df.columns[-1]
            if last_column in possible_targets:
                possible_targets.remove(last_column)
                possible_targets.insert(0, last_column)
            else:
                possible_targets.insert(0, last_column)
            
            selected_target = st.selectbox(
                "Escolha a coluna que representa a variável target:",
                ["Nenhuma"] + possible_targets,
                help="Selecione a coluna que contém a variável que você quer prever"
            )
            
            if selected_target != "Nenhuma":
                self._analyze_selected_target(df, selected_target, numeric_cols, data_source)

    def _analyze_selected_target(self, df, selected_target, numeric_cols, data_source):
        df_temp = df.copy()
        df_temp['selected_target'] = df_temp[selected_target]
        
        st.write(f"**Análise da Variável: {selected_target}**")
        
        col1, col2 = st.columns(2)
        
        with col1:
            target_counts = df_temp['selected_target'].value_counts()
            fig = px.pie(
                values=target_counts.values, 
                names=target_counts.index,
                title=f"Distribuição de {selected_target}"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(
                x=target_counts.index, 
                y=target_counts.values,
                title=f"Contagem por {selected_target}",
                labels={'x': selected_target, 'y': 'Quantidade'}
            )
            st.plotly_chart(fig, use_container_width=True)
        
        self._render_target_statistics(df_temp, selected_target, numeric_cols)
        
        if 'manual_target' not in st.session_state:
            st.session_state.manual_target = {}
        st.session_state.manual_target[data_source] = selected_target
        
        st.success(f"Variável target '{selected_target}' selecionada!")

    def _render_target_statistics(self, df_temp, selected_target, numeric_cols):
        st.write("**Estatísticas da Variável Target:**")
        col1, col2, col3 = st.columns(3)
        
        target_counts = df_temp['selected_target'].value_counts()
        
        with col1:
            st.metric("Classes Únicas", df_temp['selected_target'].nunique())
        with col2:
            st.metric("Valores Faltantes", df_temp['selected_target'].isnull().sum())
        with col3:
            most_common = target_counts.index[0]
            percentage = (target_counts.iloc[0] / len(df_temp)) * 100
            st.metric("Classe Majoritária", f"{most_common} ({percentage:.1f}%)")

    def _render_specific_visualizations(self, df, data_source, numeric_cols):
        st.write(f"**Análises Específicas - {data_source}:**")
        
        available_cols = df.columns.tolist()
        target_col = 'target_name' if 'target_name' in df.columns else 'target' if 'target' in df.columns else None
        
        if data_source == "Credit":
            self._render_credit_analysis(df, available_cols, numeric_cols, target_col)
        elif data_source == "Hipertension":
            self._render_hypertension_analysis(df, available_cols, numeric_cols, target_col)
        elif data_source == "Phone addiction":
            self._render_phone_addiction_analysis(df, available_cols, numeric_cols, target_col)
        else:
            self._render_generic_analysis(df, numeric_cols, target_col)

    def _render_credit_analysis(self, df, available_cols, numeric_cols, target_col):
        age_cols = self._find_columns(available_cols, ['age', 'idade', 'anos'])
        income_cols = self._find_columns(available_cols, ['income', 'renda', 'salary', 'salario'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            if age_cols:
                self._create_histogram(df, age_cols[0], "Distribuição de Idade")
            elif numeric_cols:
                self._create_histogram(df, numeric_cols[0], f"Distribuição de {numeric_cols[0]}")
        
        with col2:
            if income_cols:
                self._create_histogram(df, income_cols[0], f"Distribuição de {income_cols[0]}")
            elif len(numeric_cols) > 1:
                self._create_histogram(df, numeric_cols[1], f"Distribuição de {numeric_cols[1]}")
        
        if target_col and age_cols:
            self._create_age_group_analysis(df, age_cols[0], target_col)

    def _render_hypertension_analysis(self, df, available_cols, numeric_cols, target_col):
        age_cols = self._find_columns(available_cols, ['age', 'idade'])
        bp_cols = self._find_columns(available_cols, ['bp', 'pressure', 'pressao', 'systolic', 'diastolic'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            col_to_plot = age_cols[0] if age_cols else (numeric_cols[0] if numeric_cols else None)
            if col_to_plot:
                self._create_histogram(df, col_to_plot, f"Distribuição de {col_to_plot}")
        
        with col2:
            col_to_plot = bp_cols[0] if bp_cols else (numeric_cols[1] if len(numeric_cols) > 1 else None)
            if col_to_plot:
                self._create_histogram(df, col_to_plot, f"Distribuição de {col_to_plot}")
        
        if len(numeric_cols) >= 2 and target_col:
            self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1], target_col)

    def _render_phone_addiction_analysis(self, df, available_cols, numeric_cols, target_col):
        usage_cols = self._find_columns(available_cols, ['usage', 'uso', 'hours', 'horas', 'time', 'tempo'])
        sleep_cols = self._find_columns(available_cols, ['sleep', 'sono'])
        
        col1, col2 = st.columns(2)
        
        with col1:
            col_to_plot = usage_cols[0] if usage_cols else (numeric_cols[0] if numeric_cols else None)
            if col_to_plot:
                self._create_histogram(df, col_to_plot, f"Distribuição de {col_to_plot}")
        
        with col2:
            col_to_plot = sleep_cols[0] if sleep_cols else (numeric_cols[1] if len(numeric_cols) > 1 else None)
            if col_to_plot:
                self._create_histogram(df, col_to_plot, f"Distribuição de {col_to_plot}")
        
        if usage_cols and sleep_cols and target_col:
            self._create_scatter_plot(df, usage_cols[0], sleep_cols[0], target_col)

    def _render_generic_analysis(self, df, numeric_cols, target_col):
        if len(numeric_cols) >= 2:
            self._create_scatter_plot(df, numeric_cols[0], numeric_cols[1], target_col)
        
        if len(df) < 1000 and len(numeric_cols) <= 6:
            st.write("**Matriz de Scatter Plots:**")
            plot_vars = numeric_cols[:4] if len(numeric_cols) > 4 else numeric_cols
            
            if len(plot_vars) >= 2:
                try:
                    fig = px.scatter_matrix(df, dimensions=plot_vars, 
                                          color=target_col,
                                          title="Matriz de Correlações")
                    fig.update_layout(height=600)
                    st.plotly_chart(fig, use_container_width=True)
                except Exception as e:
                    st.warning(f"Não foi possível gerar a matriz de scatter plots: {str(e)}")

    def _find_columns(self, available_cols, keywords):
        return [col for col in available_cols if any(word in col.lower() for word in keywords)]

    def _create_histogram(self, df, col, title):
        fig = px.histogram(df, x=col, title=title, nbins=20)
        st.plotly_chart(fig, use_container_width=True)

    def _create_scatter_plot(self, df, x_col, y_col, color_col):
        fig = px.scatter(df, x=x_col, y=y_col, color=color_col,
                        title=f"{x_col} vs {y_col} por {color_col}")
        st.plotly_chart(fig, use_container_width=True)

    def _create_age_group_analysis(self, df, age_col, target_col):
        df_viz = df.copy()
        age_min, age_max = df[age_col].min(), df[age_col].max()
        
        if age_max - age_min > 50:
            bins = [age_min, 25, 35, 45, 55, age_max]
            labels = [f'{age_min}-25', '26-35', '36-45', '46-55', f'55-{age_max}']
            df_viz['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True)
        else:
            df_viz['age_group'] = pd.qcut(df[age_col], q=4, duplicates='drop')
        
        fig = px.histogram(df_viz, x='age_group', color=target_col, 
                         title=f"Distribuição do Target por Faixa de {age_col}",
                         barmode='group')
        st.plotly_chart(fig, use_container_width=True)

def pipeline_main():
    if "pipeline_config" not in st.session_state or st.session_state.pipeline_config is None:
        _render_welcome()
        return
    
    pipeline_config = st.session_state.pipeline_config
    data_source = pipeline_config.get("data_source")
    uploaded_file = pipeline_config.get("uploaded_file")
    
    if not data_source or (data_source == "upload" and not uploaded_file):
        _render_welcome()
        return
    
    try:
        df, nome_dataset = load_dataset(data_source, uploaded_file)
        
        if df is not None:
            st.session_state.DF = df
            
            visualizer = PipelineVisualizer()
            visualizer.render_data_overview(df, nome_dataset)
            visualizer.render_visualizations(df, data_source, nome_dataset)
            
            if 'target' in df.columns:
                _render_target_info(df)
                
        else:
            st.error("Erro ao carregar o dataset. Verifique o arquivo ou tente outro dataset.")
            _render_welcome()
            
    except Exception as e:
        st.error(f"Erro ao processar o dataset: {str(e)}")
        _render_welcome()

def _render_target_info(df):
    st.markdown("---")
    st.write("### Informações do Target")
    
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Distribuição das Classes:**")
        target_counts = df['target'].value_counts()
        
        if 'target_name' in df.columns:
            target_name_counts = df['target_name'].value_counts()
            st.write(target_name_counts)
        else:
            st.write(target_counts)
    
    with col2:
        if 'target_name' in df.columns:
            st.bar_chart(df['target_name'].value_counts())
        else:
            st.bar_chart(df['target'].value_counts())

def _render_welcome():
    st.markdown("""
    # DataVision EBAC SEMANTIX
    
    Esta aplicação permite que você configure e execute pipelines de Machine Learning 
    de forma visual e interativa.
    
    ### Como usar:
    1. **Selecione um dataset** na barra lateral
    2. **Visualize os dados** antes do processamento
    3. **Configure o algoritmo** e seus parâmetros
    4. **Execute o pipeline** e veja os resultados
    
    **Comece selecionando um dataset na barra lateral!**
    """)
    
    st.markdown("---")
    st.subheader("Datasets Disponíveis")
    
    datasets_info = {
        "Iris Dataset": {
            "problema": "Classificação de espécies de flores íris",
            "classes": "3 (Setosa, Versicolor, Virginica)",
            "features": "4 (comprimento/largura de pétalas e sépalas)",
            "amostras": "150 (50 por classe)",
            "uso": "Perfeito para iniciantes - classificação multiclasse simples"
        },
        "Wine Dataset": {
            "problema": "Classificação de vinhos por origem",
            "classes": "3 (diferentes cultivares)",
            "features": "13 (análises químicas: álcool, ácido málico, etc.)",
            "amostras": "178 vinhos",
            "uso": "Classificação com mais complexidade"
        },
        "Breast Cancer": {
            "problema": "Diagnóstico de câncer de mama",
            "classes": "2 (Maligno, Benigno)",
            "features": "30 (características dos núcleos celulares)",
            "amostras": "569 casos",
            "uso": "Classificação binária - aplicação médica importante"
        }
    }
    
    st.markdown("### Datasets Clássicos (Educacionais)")
    col1, col2, col3 = st.columns(3)
    
    cols = [col1, col2, col3]
    for i, (name, info) in enumerate(datasets_info.items()):
        with cols[i]:
            with st.expander(name):
                for key, value in info.items():
                    st.write(f"**{key.title()}**: {value}")
    
    custom_datasets_info = {
        "Credit Scoring": {
            "problema": "Análise de risco de crédito",
            "objetivo": "Prever se um cliente vai pagar o empréstimo",
            "tipo": "Classificação binária (Aprovado/Negado)",
            "aplicacao": "Bancos e fintechs"
        },
        "Hypertension": {
            "problema": "Predição de hipertensão arterial",
            "objetivo": "Identificar pacientes com risco de hipertensão",
            "tipo": "Classificação médica",
            "aplicacao": "Diagnóstico preventivo"
        },
        "Phone Addiction": {
            "problema": "Identificação de vício em smartphones",
            "objetivo": "Detectar adolescentes com uso problemático do celular",
            "tipo": "Classificação comportamental",
            "aplicacao": "Saúde mental, bem-estar digital"
        }
    }
    
    st.markdown("### Datasets Personalizados (Projetos Reais)")
    col1, col2, col3 = st.columns(3)
    
    cols = [col1, col2, col3]
    for i, (name, info) in enumerate(custom_datasets_info.items()):
        with cols[i]:
            with st.expander(name):
                for key, value in info.items():
                    st.write(f"**{key.title()}**: {value}")
    
    if 'executions' in st.session_state and st.session_state.executions:
        st.markdown("---")
        st.subheader("Histórico de Execuções")
        
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