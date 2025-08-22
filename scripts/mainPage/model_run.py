import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

def model_run():
    st.header("Resultados do Pipeline")
    
    if 'executions' not in st.session_state or not st.session_state.executions:
        st.warning("Nenhuma execução de pipeline encontrada!")
        st.info("Execute um pipeline primeiro na seção 'Pipeline de Dados'.")
        
        if st.button("Voltar ao Pipeline"):
            st.session_state.categoria = "Pipeline"
            st.rerun()
        return
    
    latest_execution = st.session_state.executions[-1]
    
    render_execution_summary(latest_execution)
    render_detailed_analysis(latest_execution)
    render_execution_history()

def render_execution_summary(execution):
    st.subheader("Resumo da Última Execução")
    
    config = execution['config']
    results = execution['results']
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Dataset", config.get('data_source', 'N/A'))
        st.metric("Algoritmo", config.get('algorithm', 'N/A'))
    
    with col2:
        st.metric("Status", results.get('status', 'unknown'))
        timestamp = execution['timestamp'].strftime('%H:%M:%S')
        st.metric("Horário", timestamp)
    
    with col3:
        if 'evaluation' in results and 'metrics' in results['evaluation']:
            metrics = results['evaluation']['metrics']
            accuracy = metrics.get('accuracy', 0)
            st.metric("Accuracy", f"{accuracy:.4f}")
    
    with col4:
        if 'evaluation' in results and 'metrics' in results['evaluation']:
            metrics = results['evaluation']['metrics']
            f1 = metrics.get('f1_score', 0)
            st.metric("F1-Score", f"{f1:.4f}")

def render_detailed_analysis(execution):
    st.markdown("---")
    st.subheader("Análise Detalhada")
    
    config = execution['config']
    results = execution['results']
    
    tab1, tab2, tab3 = st.tabs(["Métricas", "Configuração", "Recomendações"])
    
    with tab1:
        render_metrics_analysis(results)
    
    with tab2:
        render_configuration_details(config)
    
    with tab3:
        render_recommendations(config, results)

def render_metrics_analysis(results):
    if 'evaluation' not in results or 'metrics' not in results['evaluation']:
        st.warning("Métricas de avaliação não encontradas.")
        return
    
    metrics = results['evaluation']['metrics']
    
    st.write("**Métricas de Performance:**")
    
    metric_names = ['accuracy', 'precision', 'recall', 'f1_score']
    metric_values = [metrics.get(name, 0) for name in metric_names]
    metric_labels = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = go.Figure(data=[
            go.Bar(x=metric_labels, y=metric_values, 
                  text=[f"{v:.3f}" for v in metric_values],
                  textposition='auto')
        ])
        fig.update_layout(
            title="Métricas de Avaliação",
            yaxis_title="Score",
            yaxis=dict(range=[0, 1])
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = go.Figure(data=go.Scatterpolar(
            r=metric_values,
            theta=metric_labels,
            fill='toself',
            name='Performance'
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            title="Radar de Performance"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    if 'cv_mean' in metrics and 'cv_std' in metrics:
        st.write("**Validação Cruzada:**")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("CV Mean", f"{metrics['cv_mean']:.4f}")
        with col2:
            st.metric("CV Std", f"{metrics['cv_std']:.4f}")

def render_configuration_details(config):
    st.write("**Configurações do Pipeline:**")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Configurações Básicas:**")
        basic_config = {
            "Data Source": config.get('data_source', 'N/A'),
            "Algorithm": config.get('algorithm', 'N/A'),
            "Test Size": config.get('test_size', 'N/A'),
            "Scaling": config.get('scaling', 'N/A')
        }
        
        for key, value in basic_config.items():
            st.write(f"- **{key}**: {value}")
    
    with col2:
        st.write("**Configurações Avançadas:**")
        advanced_config = {
            "Feature Selection": config.get('feature_selection', False),
            "Balance Classes": config.get('balance_classes', False),
            "Tune Hyperparameters": config.get('tune_hyperparameters', False),
            "Cross Validation": config.get('use_cv', False)
        }
        
        for key, value in advanced_config.items():
            status = "Ativado" if value else "Desativado"
            st.write(f"- **{key}**: {status}")
    
    if 'algo_params' in config:
        st.write("**Parâmetros do Algoritmo:**")
        st.json(config['algo_params'])

def render_recommendations(config, results):
    st.write("**Recomendações para Melhorar a Performance:**")
    
    recommendations = []
    
    if 'evaluation' in results and 'metrics' in results['evaluation']:
        accuracy = results['evaluation']['metrics'].get('accuracy', 0)
        
        if accuracy < 0.7:
            recommendations.append({
                "type": "error",
                "title": "Accuracy Baixa",
                "description": "Considere coletar mais dados, revisar a qualidade dos dados ou testar outros algoritmos."
            })
        elif accuracy < 0.8:
            recommendations.append({
                "type": "warning",
                "title": "Accuracy Moderada",
                "description": "Tente feature engineering, otimização de hiperparâmetros ou balanceamento de classes."
            })
        else:
            recommendations.append({
                "type": "success",
                "title": "Boa Performance",
                "description": "Modelo está performando bem! Considere validação adicional com dados externos."
            })
    
    if not config.get('tune_hyperparameters', False):
        recommendations.append({
            "type": "info",
            "title": "Otimização de Hiperparâmetros",
            "description": "Ative a otimização automática de hiperparâmetros para melhorar a performance."
        })
    
    if not config.get('feature_selection', False):
        recommendations.append({
            "type": "info",
            "title": "Seleção de Features",
            "description": "Considere ativar a seleção automática de features para reduzir overfitting."
        })
    
    if not config.get('balance_classes', False):
        recommendations.append({
            "type": "info",
            "title": "Balanceamento de Classes",
            "description": "Se o dataset estiver desbalanceado, ative o balanceamento de classes."
        })
    
    for rec in recommendations:
        if rec["type"] == "error":
            st.error(f"**{rec['title']}**: {rec['description']}")
        elif rec["type"] == "warning":
            st.warning(f"**{rec['title']}**: {rec['description']}")
        elif rec["type"] == "success":
            st.success(f"**{rec['title']}**: {rec['description']}")
        else:
            st.info(f"**{rec['title']}**: {rec['description']}")

def render_execution_history():
    st.markdown("---")
    st.subheader("Histórico de Execuções")
    
    if len(st.session_state.executions) == 0:
        st.info("Nenhuma execução anterior encontrada.")
        return
    
    executions_data = []
    for i, execution in enumerate(st.session_state.executions):
        config = execution['config']
        results = execution['results']
        
        accuracy = 0
        if 'evaluation' in results and 'metrics' in results['evaluation']:
            accuracy = results['evaluation']['metrics'].get('accuracy', 0)
        
        executions_data.append({
            "Execução": i + 1,
            "Timestamp": execution['timestamp'].strftime('%d/%m/%Y %H:%M:%S'),
            "Dataset": config.get('data_source', 'N/A'),
            "Algoritmo": config.get('algorithm', 'N/A'),
            "Accuracy": f"{accuracy:.4f}",
            "Status": results.get('status', 'unknown')
        })
    
    df_executions = pd.DataFrame(executions_data)
    st.dataframe(df_executions, use_container_width=True)
    
    if len(executions_data) > 1:
        render_execution_comparison(executions_data)

def render_execution_comparison(executions_data):
    st.write("**Comparação de Performance:**")
    
    df = pd.DataFrame(executions_data)
    df['Accuracy_Float'] = df['Accuracy'].astype(float)
    
    fig = px.line(df, x='Execução', y='Accuracy_Float', 
                  title='Evolução da Accuracy ao Longo das Execuções',
                  markers=True)
    fig.update_layout(yaxis_title='Accuracy')
    st.plotly_chart(fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        best_execution = df.loc[df['Accuracy_Float'].idxmax()]
        st.metric("Melhor Execução", 
                 f"#{int(best_execution['Execução'])}", 
                 f"Accuracy: {best_execution['Accuracy']}")
    
    with col2:
        avg_accuracy = df['Accuracy_Float'].mean()
        st.metric("Accuracy Média", f"{avg_accuracy:.4f}")
    
    st.write("**Comparação por Algoritmo:**")
    algorithm_performance = df.groupby('Algoritmo')['Accuracy_Float'].agg(['mean', 'max', 'count']).round(4)
    algorithm_performance.columns = ['Accuracy Média', 'Melhor Accuracy', 'Execuções']
    st.dataframe(algorithm_performance, use_container_width=True)