import streamlit as st
import pandas as pd
from scripts.data_loader import load_dataset

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
            
            st.write(f"### Dataset: {nome_dataset}")
            
            tab1, tab2, tab3 = st.tabs(["Prévia dos Dados", "Estatísticas", "Informações Gerais"])
            
            with tab1:
                st.write("**Primeiras 5 linhas:**")
                st.dataframe(df.head(5), use_container_width=True)
            
            with tab2:
                st.write("**Estatísticas descritivas:**")
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
                    st.metric("Linhas", df.shape[0])
                    st.metric("Colunas", df.shape[1])
                    st.metric("Tamanho em Memória", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
                
                with col2:
                    missing_count = df.isnull().sum().sum()
                    missing_percent = (missing_count / (df.shape[0] * df.shape[1])) * 100
                    
                    st.metric("Valores Faltantes", f"{missing_count} ({missing_percent:.1f}%)")
                    st.metric("Colunas Numéricas", len(numeric_cols))
                    st.metric("Colunas Categóricas", len(categorical_cols))
                
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
            
            _render_dataset_visualizations(df, data_source, nome_dataset)
            
            if 'target' in df.columns:
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
        else:
            st.error("Erro ao carregar o dataset. Verifique o arquivo ou tente outro dataset.")
            _render_welcome()
            
    except Exception as e:
        st.error(f"Erro ao processar o dataset: {str(e)}")
        _render_welcome()


def _render_dataset_visualizations(df, data_source, nome_dataset):
    import matplotlib.pyplot as plt
    import seaborn as sns
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    
    plt.style.use('seaborn-v0_8')
    
    st.markdown("---")
    st.subheader("Visualizações do Dataset")
    
    has_target = 'target' in df.columns or 'target_name' in df.columns
    
    manual_target_selected = False
    if 'manual_target' in st.session_state and data_source in st.session_state.manual_target:
        manual_target_col = st.session_state.manual_target[data_source]
        if manual_target_col in df.columns:
            has_target = True
            manual_target_selected = True
    
    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
    if 'target' in numeric_cols:
        numeric_cols.remove('target')
    
    viz_tabs = st.tabs(["Correlações", "Distribuições", "Target Analysis", "Específicas do Dataset"])
    
    with viz_tabs[0]:
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
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Não há colunas numéricas suficientes para matriz de correlação")
    
    with viz_tabs[1]:
        if len(numeric_cols) > 0:
            st.write("**Distribuições das Variáveis Numéricas:**")
            
            cols_to_plot = st.multiselect(
                "Selecione as colunas para visualizar:",
                numeric_cols,
                default=numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            )
            
            if cols_to_plot:
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
    
    with viz_tabs[2]:
        if has_target:
            st.write("**Análise da Variável Target:**")
            
            if manual_target_selected:
                target_col = st.session_state.manual_target[data_source]
                col_info, col_reset = st.columns([3, 1])
                with col_info:
                    st.info(f"Usando '{target_col}' como variável target (seleção manual)")
                with col_reset:
                    if st.button("Alterar", help="Clique para selecionar outra coluna target"):
                        del st.session_state.manual_target[data_source]
                        st.rerun()
            else:
                target_col = 'target_name' if 'target_name' in df.columns else 'target'
            
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
            st.write("**Selecione a Variável Target:**")
            
            possible_targets = []
            for col in df.columns:
                unique_values = df[col].nunique()
                if 2 <= unique_values <= 10:
                    possible_targets.append(col)
            
            # Colocar a última coluna como primeira opção
            if possible_targets:
                last_column = df.columns[-1]
                if last_column in possible_targets:
                    possible_targets.remove(last_column)
                    possible_targets.insert(0, last_column)
                else:
                    # Se a última coluna não está nos targets possíveis, adicionar ela mesmo assim
                    possible_targets.insert(0, last_column)
                
                selected_target = st.selectbox(
                    "Escolha a coluna que representa a variável target:",
                    ["Nenhuma"] + possible_targets,
                    help="Selecione a coluna que contém a variável que você quer prever"
                )
                
                if selected_target != "Nenhuma":
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
                    
                    if len(numeric_cols) > 0 and df_temp['selected_target'].dtype in ['int64', 'float64']:
                        st.write("**Correlação com Variáveis Numéricas:**")
                        
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
                    
                    if 'manual_target' not in st.session_state:
                        st.session_state.manual_target = {}
                    st.session_state.manual_target[data_source] = selected_target
                    
                    st.success(f"Variável target '{selected_target}' selecionada! Esta informação será lembrada para este dataset.")
                
                else:
                    st.info("Selecione uma coluna target para ver as análises")
            else:
                st.warning("Não foram encontradas colunas adequadas para serem target (colunas com 2-10 valores únicos)")
                st.info("Dica: Variáveis target geralmente têm poucos valores únicos (ex: classes, categorias)")
    
    with viz_tabs[3]:
        _render_specific_visualizations(df, data_source, numeric_cols)


def _render_specific_visualizations(df, data_source, numeric_cols):
    """
    Renderiza visualizações específicas baseadas no dataset real carregado
    """
    import plotly.express as px
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    import pandas as pd
    
    st.write(f"**Análises Específicas - {data_source}:**")
    
    # Verificar colunas disponíveis no dataset
    available_cols = df.columns.tolist()
    
    if data_source == "Credit":
        st.write("**Análise de Credit Scoring:**")
        
        # Detectar automaticamente colunas relevantes
        age_cols = [col for col in available_cols if any(word in col.lower() for word in ['age', 'idade', 'anos'])]
        income_cols = [col for col in available_cols if any(word in col.lower() for word in ['income', 'renda', 'salary', 'salario'])]
        credit_cols = [col for col in available_cols if any(word in col.lower() for word in ['credit', 'score', 'credito'])]
        amount_cols = [col for col in available_cols if any(word in col.lower() for word in ['amount', 'valor', 'loan', 'emprestimo'])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Idade
            if age_cols:
                age_col = age_cols[0]
                fig = px.histogram(df, x=age_col, title=f"Distribuição de {age_col}", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            elif 'age' in df.columns:
                fig = px.histogram(df, x='age', title="Distribuição de Idade", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            else:
                # Usar primeira coluna numérica como proxy
                if numeric_cols:
                    fig = px.histogram(df, x=numeric_cols[0], title=f"Distribuição de {numeric_cols[0]}", nbins=20)
                    st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Renda/Income
            if income_cols:
                income_col = income_cols[0]
                fig = px.histogram(df, x=income_col, title=f"Distribuição de {income_col}", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            elif credit_cols:
                credit_col = credit_cols[0]
                fig = px.histogram(df, x=credit_col, title=f"Distribuição de {credit_col}", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
            elif len(numeric_cols) > 1:
                fig = px.histogram(df, x=numeric_cols[1], title=f"Distribuição de {numeric_cols[1]}", nbins=20)
                st.plotly_chart(fig, use_container_width=True)
        
        # Análise por faixa etária se tiver target
        target_col = 'target_name' if 'target_name' in df.columns else 'target' if 'target' in df.columns else None
        
        if target_col and age_cols:
            age_col = age_cols[0]
            df_viz = df.copy()
            
            # Criar faixas etárias baseadas nos dados reais
            age_min, age_max = df[age_col].min(), df[age_col].max()
            if age_max - age_min > 50:
                bins = [age_min, 25, 35, 45, 55, age_max]
                labels = [f'{age_min}-25', '26-35', '36-45', '46-55', f'55-{age_max}']
            else:
                # Se range menor, criar bins automáticos
                bins = pd.qcut(df[age_col], q=4, duplicates='drop').cat.categories
                labels = [f'{int(interval.left)}-{int(interval.right)}' for interval in bins]
                bins = pd.qcut(df[age_col], q=4, duplicates='drop')
            
            if isinstance(bins, list):
                df_viz['age_group'] = pd.cut(df[age_col], bins=bins, labels=labels, include_lowest=True)
            else:
                df_viz['age_group'] = bins
            
            fig = px.histogram(df_viz, x='age_group', color=target_col, 
                             title=f"Distribuição do Target por Faixa de {age_col}",
                             barmode='group')
            st.plotly_chart(fig, use_container_width=True)
        
        # Análise de correlação com variáveis numéricas relevantes
        if len(numeric_cols) >= 2:
            st.write("**Correlação entre Variáveis Principais:**")
            
            # Selecionar as 4 variáveis mais relevantes
            relevant_cols = numeric_cols[:4] if len(numeric_cols) >= 4 else numeric_cols
            
            fig = px.scatter_matrix(df, dimensions=relevant_cols, 
                                  color=target_col if target_col else None,
                                  title="Matriz de Correlações - Principais Variáveis")
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
    
    elif data_source == "Hipertension":
        st.write("**Análise de Hipertensão:**")
        
        # Detectar colunas relevantes
        age_cols = [col for col in available_cols if any(word in col.lower() for word in ['age', 'idade'])]
        bp_cols = [col for col in available_cols if any(word in col.lower() for word in ['bp', 'pressure', 'pressao', 'systolic', 'diastolic'])]
        bmi_cols = [col for col in available_cols if any(word in col.lower() for word in ['bmi', 'weight', 'height', 'peso', 'altura'])]
        gender_cols = [col for col in available_cols if any(word in col.lower() for word in ['gender', 'sex', 'genero', 'sexo'])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Idade
            if age_cols:
                age_col = age_cols[0]
                fig = px.histogram(df, x=age_col, title=f"Distribuição de {age_col}", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
            elif numeric_cols:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribuição de {numeric_cols[0]}", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # BMI ou segunda variável numérica
            if bmi_cols:
                bmi_col = bmi_cols[0]
                fig = px.histogram(df, x=bmi_col, title=f"Distribuição de {bmi_col}", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
            elif len(numeric_cols) > 1:
                fig = px.histogram(df, x=numeric_cols[1], title=f"Distribuição de {numeric_cols[1]}", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot de pressão arterial se disponível
        systolic_col = next((col for col in bp_cols if 'systolic' in col.lower() or 'sistolica' in col.lower()), None)
        diastolic_col = next((col for col in bp_cols if 'diastolic' in col.lower() or 'diastolica' in col.lower()), None)
        target_col = 'target_name' if 'target_name' in df.columns else 'target' if 'target' in df.columns else None
        
        if systolic_col and diastolic_col and target_col:
            fig = px.scatter(df, x=systolic_col, y=diastolic_col, color=target_col,
                           title=f"{systolic_col} vs {diastolic_col} por Diagnóstico")
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=target_col,
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]} por Diagnóstico")
            st.plotly_chart(fig, use_container_width=True)
        
        # Análise de fatores de risco (colunas binárias)
        binary_cols = []
        for col in available_cols:
            if df[col].dtype in ['int64', 'float64'] and df[col].nunique() == 2:
                unique_vals = sorted(df[col].unique())
                if unique_vals == [0, 1] or unique_vals == [0.0, 1.0]:
                    binary_cols.append(col)
        
        if binary_cols:
            st.write("**Análise de Fatores de Risco (Variáveis Binárias):**")
            
            factor_data = []
            for factor in binary_cols[:6]:  # Limitar a 6 fatores para não poluir
                positive_rate = (df[factor] == 1).mean() * 100
                factor_name = factor.replace('_', ' ').replace('-', ' ').title()
                factor_data.append({'Fator': factor_name, 'Taxa (%)': positive_rate})
            
            if factor_data:
                factor_df = pd.DataFrame(factor_data)
                fig = px.bar(factor_df, x='Fator', y='Taxa (%)', 
                           title="Taxa de Fatores de Risco na População")
                fig.update_xaxis(tickangle=45)
                st.plotly_chart(fig, use_container_width=True)
    
    elif data_source == "Phone addiction":
        st.write("**Análise de Vício em Smartphone:**")
        
        # Detectar colunas relevantes
        usage_cols = [col for col in available_cols if any(word in col.lower() for word in ['usage', 'uso', 'hours', 'horas', 'time', 'tempo'])]
        sleep_cols = [col for col in available_cols if any(word in col.lower() for word in ['sleep', 'sono'])]
        age_cols = [col for col in available_cols if any(word in col.lower() for word in ['age', 'idade'])]
        social_cols = [col for col in available_cols if any(word in col.lower() for word in ['social', 'apps', 'media'])]
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Uso diário
            if usage_cols:
                usage_col = usage_cols[0]
                fig = px.histogram(df, x=usage_col, title=f"Distribuição de {usage_col}", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
            elif numeric_cols:
                fig = px.histogram(df, x=numeric_cols[0], title=f"Distribuição de {numeric_cols[0]}", nbins=15)
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Sono
            if sleep_cols:
                sleep_col = sleep_cols[0]
                fig = px.histogram(df, x=sleep_col, title=f"Distribuição de {sleep_col}", nbins=10)
                st.plotly_chart(fig, use_container_width=True)
            elif len(numeric_cols) > 1:
                fig = px.histogram(df, x=numeric_cols[1], title=f"Distribuição de {numeric_cols[1]}", nbins=10)
                st.plotly_chart(fig, use_container_width=True)
        
        # Correlação uso vs sono
        target_col = 'target_name' if 'target_name' in df.columns else 'target' if 'target' in df.columns else None
        
        if usage_cols and sleep_cols:
            usage_col, sleep_col = usage_cols[0], sleep_cols[0]
            fig = px.scatter(df, x=usage_col, y=sleep_col, color=target_col,
                           title=f"Relação entre {usage_col} e {sleep_col}")
            st.plotly_chart(fig, use_container_width=True)
        elif len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=target_col,
                           title=f"Relação entre {numeric_cols[0]} e {numeric_cols[1]}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Análise por idade se disponível
        if age_cols and usage_cols:
            age_col, usage_col = age_cols[0], usage_cols[0]
            fig = px.box(df, x=age_col, y=usage_col, 
                        title=f"{usage_col} por {age_col}")
            st.plotly_chart(fig, use_container_width=True)
        elif age_cols and numeric_cols:
            age_col = age_cols[0]
            fig = px.box(df, x=age_col, y=numeric_cols[0], 
                        title=f"{numeric_cols[0]} por {age_col}")
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        # Para datasets genéricos ou desconhecidos
        st.write(f"**Análises Gerais - {data_source.title()}:**")
        
        target_col = 'target_name' if 'target_name' in df.columns else 'target' if 'target' in df.columns else None
        
        # Scatter plot das duas primeiras variáveis numéricas
        if len(numeric_cols) >= 2:
            fig = px.scatter(df, x=numeric_cols[0], y=numeric_cols[1], color=target_col,
                           title=f"{numeric_cols[0]} vs {numeric_cols[1]}")
            st.plotly_chart(fig, use_container_width=True)
        
        # Matriz de scatter plots se dataset não for muito grande
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
        
        # Análise das variáveis categóricas se houver
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if 'target_name' in categorical_cols:
            categorical_cols.remove('target_name')
        if 'target' in categorical_cols:
            categorical_cols.remove('target')
        
        if categorical_cols and target_col:
            st.write("**Análise de Variáveis Categóricas:**")
            
            # Mostrar apenas as primeiras 3 variáveis categóricas
            for cat_col in categorical_cols[:3]:
                if df[cat_col].nunique() <= 10:  # Só mostrar se não tiver muitas categorias
                    fig = px.histogram(df, x=cat_col, color=target_col,
                                     title=f"Distribuição de {cat_col} por Target",
                                     barmode='group')
                    fig.update_xaxis(tickangle=45)
                    st.plotly_chart(fig, use_container_width=True)
    
    # Informações adicionais sobre o dataset
    st.markdown("---")
    st.write("**Informações do Dataset:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total de Colunas", len(df.columns))
        st.metric("Colunas Numéricas", len(numeric_cols))
    
    with col2:
        categorical_count = len(df.select_dtypes(include=['object', 'category']).columns)
        st.metric("Colunas Categóricas", categorical_count)
        missing_percent = (df.isnull().sum().sum() / (df.shape[0] * df.shape[1])) * 100
        st.metric("% Dados Faltantes", f"{missing_percent:.1f}%")
    
    with col3:
        memory_usage = df.memory_usage(deep=True).sum() / 1024 / 1024  # MB
        st.metric("Uso de Memória", f"{memory_usage:.1f} MB")
        
        if target_col:
            balance_ratio = df[target_col].value_counts().min() / df[target_col].value_counts().max()
            st.metric("Balanceamento", f"{balance_ratio:.2f}")
    
    # Lista das colunas disponíveis para debug
    with st.expander("🔍 Colunas Disponíveis no Dataset (Debug)"):
        st.write("**Todas as colunas:**")
        cols_info = []
        for col in df.columns:
            dtype = str(df[col].dtype)
            unique_count = df[col].nunique()
            missing_count = df[col].isnull().sum()
            cols_info.append({
                'Coluna': col,
                'Tipo': dtype,
                'Valores Únicos': unique_count,
                'Faltantes': missing_count
            })
        
        cols_df = pd.DataFrame(cols_info)
        st.dataframe(cols_df, use_container_width=True)
        
        st.write("**Primeiras linhas para referência:**")
        st.dataframe(df.head(3), use_container_width=True)


def _render_welcome():
    st.markdown("""
    # DataVision EBAC SEMANTIX
    
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
    
    st.markdown("---")
    st.subheader("Datasets Disponíveis")
    st.markdown("Conheça os datasets que você pode usar nesta aplicação:")
    
    st.markdown("### Datasets Clássicos (Educacionais)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("Iris Dataset"):
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
        with st.expander("Wine Dataset"):
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
        with st.expander("Breast Cancer"):
            st.markdown("""
            **Breast Cancer Dataset**
            - **Problema**: Diagnóstico de câncer de mama
            - **Classes**: 2 (Maligno, Benigno)
            - **Features**: 30 (características dos núcleos celulares)
            - **Amostras**: 569 casos
            - **Uso**: Classificação binária - aplicação médica importante
            - **Origem**: Hospital da Universidade de Wisconsin
            """)
    
    st.markdown("### Datasets Personalizados (Projetos Reais)")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        with st.expander("Credit Scoring"):
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
        with st.expander("Hypertension"):
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
        with st.expander("Phone Addiction"):
            st.markdown("""
            **Teen Phone Addiction Dataset**
            - **Problema**: Identificação de vício em smartphones
            - **Objetivo**: Detectar adolescentes com uso problemático do celular
            - **Tipo**: Classificação comportamental
            - **Aplicação**: Saúde mental, bem-estar digital
            - **Importância**: Problema crescente na era digital
            - **Features**: Padrões de uso, comportamento, dados psicológicos
            """)
    
    st.markdown("---")
    st.markdown("### Recursos da Aplicação")
    
    feature_col1, feature_col2 = st.columns(2)
    
    with feature_col1:
        st.markdown("""
        **Análise de Dados:**
        - Prévia interativa dos datasets
        - Estatísticas descritivas automáticas
        - Visualizações exploratórias
        - Análise de qualidade dos dados
        - Detecção de valores faltantes
        - Análise de correlações
        """)
    
    with feature_col2:
        st.markdown("""
        **Machine Learning:**
        - Múltiplos algoritmos (RF, SVM, LogReg)
        - Configuração de hiperparâmetros
        - Cross-validation automática
        - Métricas de avaliação completas
        - Visualizações de resultados
        - Histórico de experimentos
        """)
    
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