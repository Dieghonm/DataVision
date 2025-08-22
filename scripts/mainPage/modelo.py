import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os
from datetime import datetime
import plotly.express as px

def model_main():
    st.header("Gerenciamento de Modelos Salvos")
    
    tab1, tab2, tab3 = st.tabs(["Carregar Modelo", "Fazer Predições", "Análise de Modelos"])
    
    with tab1:
        load_saved_model()
    
    with tab2:
        make_predictions()
    
    with tab3:
        analyze_models()

def load_saved_model():
    st.subheader("Carregar Modelo Salvo")
    
    models_dir = "data/models"
    results_dir = "data/results"
    
    if not os.path.exists(models_dir) or not os.path.exists(results_dir):
        st.warning("Diretórios de modelos não encontrados. Execute um pipeline primeiro.")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not model_files:
        st.info("Nenhum modelo salvo encontrado. Execute um pipeline primeiro.")
        return
    
    selected_model = st.selectbox(
        "Selecione um modelo:",
        model_files,
        format_func=lambda x: x.replace('.pkl', '').replace('model_', 'Modelo ')
    )
    
    if selected_model:
        model_path = os.path.join(models_dir, selected_model)
        result_file = selected_model.replace('.pkl', '.json').replace('model_', 'results_')
        result_path = os.path.join(results_dir, result_file)
        
        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)
            
            if os.path.exists(result_path):
                with open(result_path, 'r') as f:
                    results = json.load(f)
                
                display_model_info(model, results, selected_model)
                
                if st.button("Carregar Este Modelo", type="primary"):
                    st.session_state.loaded_model = model
                    st.session_state.loaded_model_results = results
                    st.session_state.loaded_model_name = selected_model
                    st.success(f"Modelo {selected_model} carregado com sucesso!")
            else:
                st.warning("Arquivo de resultados não encontrado para este modelo.")
                
        except Exception as e:
            st.error(f"Erro ao carregar modelo: {str(e)}")

def display_model_info(model, results, model_name):
    st.write(f"**Informações do Modelo: {model_name}**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Configurações:**")
        config = results.get('config', {})
        st.write(f"- Algoritmo: {config.get('algorithm', 'N/A')}")
        st.write(f"- Dataset: {config.get('data_source', 'N/A')}")
        st.write(f"- Test Size: {config.get('test_size', 'N/A')}")
    
    with col2:
        st.write("**Performance:**")
        eval_results = results.get('evaluation_results', {})
        st.write(f"- Accuracy: {eval_results.get('accuracy', 0):.4f}")
        st.write(f"- F1-Score: {eval_results.get('f1_score', 0):.4f}")
        st.write(f"- Precision: {eval_results.get('precision', 0):.4f}")
    
    with col3:
        st.write("**Detalhes:**")
        data_shape = results.get('data_shape', [0, 0])
        st.write(f"- Amostras: {data_shape[0]}")
        st.write(f"- Features: {data_shape[1]}")
        st.write(f"- Timestamp: {results.get('timestamp', 'N/A')}")
    
    if 'selected_features' in results:
        features = results['selected_features']
        if features:
            st.write("**Features Selecionadas:**")
            st.write(f"- Total: {len(features)}")
            st.write(f"- Features: {', '.join(features[:5])}{'...' if len(features) > 5 else ''}")
    
    if 'best_params' in results and results['best_params']:
        st.write("**Melhores Parâmetros:**")
        st.json(results['best_params'])

def make_predictions():
    st.subheader("Fazer Predições")
    
    if 'loaded_model' not in st.session_state:
        st.warning("Carregue um modelo primeiro na aba 'Carregar Modelo'.")
        return
    
    model = st.session_state.loaded_model
    results = st.session_state.loaded_model_results
    model_name = st.session_state.loaded_model_name
    
    st.success(f"Modelo carregado: {model_name}")
    
    st.write("**Opções de Predição:**")
    prediction_mode = st.radio(
        "Como deseja fazer as predições?",
        ["Upload de arquivo CSV", "Entrada manual de dados", "Usar dados de exemplo"]
    )
    
    if prediction_mode == "Upload de arquivo CSV":
        make_batch_predictions(model, results)
    elif prediction_mode == "Entrada manual de dados":
        make_manual_prediction(model, results)
    else:
        make_example_predictions(model, results)

def make_batch_predictions(model, results):
    st.write("**Predições em Lote (CSV):**")
    
    uploaded_file = st.file_uploader(
        "Faça upload do arquivo CSV com os dados para predição:",
        type=['csv'],
        help="O arquivo deve conter as mesmas colunas usadas no treinamento"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("**Dados carregados:**")
            st.dataframe(df.head(), use_container_width=True)
            
            if st.button("Fazer Predições", type="primary"):
                predictions = model.predict(df)
                probabilities = None
                
                if hasattr(model, 'predict_proba'):
                    try:
                        probabilities = model.predict_proba(df)
                    except:
                        pass
                
                df_results = df.copy()
                df_results['Predição'] = predictions
                
                if probabilities is not None:
                    for i in range(probabilities.shape[1]):
                        df_results[f'Probabilidade_Classe_{i}'] = probabilities[:, i]
                
                st.write("**Resultados das Predições:**")
                st.dataframe(df_results, use_container_width=True)
                
                csv = df_results.to_csv(index=False)
                st.download_button(
                    label="Download dos Resultados (CSV)",
                    data=csv,
                    file_name=f"predicoes_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
                
        except Exception as e:
            st.error(f"Erro ao processar arquivo: {str(e)}")

def make_manual_prediction(model, results):
    st.write("**Entrada Manual de Dados:**")
    
    selected_features = results.get('selected_features', [])
    data_shape = results.get('data_shape', [0, 0])
    
    if selected_features:
        features_to_use = selected_features
    else:
        features_to_use = [f"feature_{i}" for i in range(data_shape[1] - 1)]
    
    st.write(f"**Digite os valores para as {len(features_to_use)} features:**")
    
    input_data = {}
    cols = st.columns(min(3, len(features_to_use)))
    
    for i, feature in enumerate(features_to_use):
        col_idx = i % len(cols)
        with cols[col_idx]:
            input_data[feature] = st.number_input(
                f"{feature}:",
                value=0.0,
                step=0.01,
                key=f"manual_{feature}"
            )
    
    if st.button("Fazer Predição", type="primary"):
        try:
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)[0]
            
            st.success(f"Predição: {prediction}")
            
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(input_df)[0]
                    st.write("**Probabilidades por Classe:**")
                    
                    prob_data = []
                    for i, prob in enumerate(probabilities):
                        prob_data.append({"Classe": i, "Probabilidade": prob})
                    
                    prob_df = pd.DataFrame(prob_data)
                    
                    fig = px.bar(prob_df, x='Classe', y='Probabilidade',
                                title='Probabilidades de Cada Classe')
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Não foi possível calcular probabilidades: {str(e)}")
                    
        except Exception as e:
            st.error(f"Erro ao fazer predição: {str(e)}")

def make_example_predictions(model, results):
    st.write("**Predições com Dados de Exemplo:**")
    
    config = results.get('config', {})
    data_source = config.get('data_source', '')
    
    if data_source in ['iris', 'wine', 'breast_cancer']:
        make_sklearn_example_predictions(model, results, data_source)
    else:
        st.info("Dados de exemplo não disponíveis para este dataset.")
        st.write("Use a opção 'Entrada manual de dados' ou 'Upload de arquivo CSV'.")

def make_sklearn_example_predictions(model, results, data_source):
    from sklearn.datasets import load_iris, load_wine, load_breast_cancer
    
    dataset_map = {
        'iris': load_iris,
        'wine': load_wine,
        'breast_cancer': load_breast_cancer
    }
    
    if data_source not in dataset_map:
        st.warning("Dataset de exemplo não encontrado.")
        return
    
    data = dataset_map[data_source]()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y_true = data.target
    
    n_samples = st.slider("Número de amostras para testar:", 5, 50, 10)
    
    sample_indices = np.random.choice(len(X), n_samples, replace=False)
    X_sample = X.iloc[sample_indices]
    y_sample = y_true[sample_indices]
    
    if st.button("Fazer Predições de Exemplo", type="primary"):
        try:
            predictions = model.predict(X_sample)
            
            results_df = X_sample.copy()
            results_df['Valor_Real'] = y_sample
            results_df['Predição'] = predictions
            results_df['Correto'] = results_df['Valor_Real'] == results_df['Predição']
            
            st.write("**Resultados das Predições:**")
            st.dataframe(results_df, use_container_width=True)
            
            accuracy = (results_df['Correto'].sum() / len(results_df)) * 100
            st.metric("Accuracy na Amostra", f"{accuracy:.1f}%")
            
            if hasattr(model, 'predict_proba'):
                try:
                    probabilities = model.predict_proba(X_sample)
                    
                    st.write("**Distribuição de Confiança das Predições:**")
                    max_probs = np.max(probabilities, axis=1)
                    
                    fig = px.histogram(max_probs, nbins=10, 
                                     title='Distribuição da Confiança Máxima',
                                     labels={'value': 'Confiança Máxima', 'count': 'Frequência'})
                    st.plotly_chart(fig, use_container_width=True)
                    
                except Exception as e:
                    st.warning(f"Não foi possível calcular probabilidades: {str(e)}")
                    
        except Exception as e:
            st.error(f"Erro ao fazer predições: {str(e)}")

def analyze_models():
    st.subheader("Análise Comparativa de Modelos")
    
    models_dir = "data/models"
    results_dir = "data/results"
    
    if not os.path.exists(results_dir):
        st.warning("Diretório de resultados não encontrado.")
        return
    
    result_files = [f for f in os.listdir(results_dir) if f.endswith('.json')]
    
    if not result_files:
        st.info("Nenhum resultado de modelo encontrado.")
        return
    
    models_data = []
    
    for result_file in result_files:
        try:
            with open(os.path.join(results_dir, result_file), 'r') as f:
                results = json.load(f)
            
            config = results.get('config', {})
            eval_results = results.get('evaluation_results', {})
            
            models_data.append({
                'Modelo': result_file.replace('.json', '').replace('results_', ''),
                'Algoritmo': config.get('algorithm', 'N/A'),
                'Dataset': config.get('data_source', 'N/A'),
                'Accuracy': eval_results.get('accuracy', 0),
                'F1-Score': eval_results.get('f1_score', 0),
                'Precision': eval_results.get('precision', 0),
                'Recall': eval_results.get('recall', 0),
                'Timestamp': results.get('timestamp', 'N/A'),
                'Test_Size': config.get('test_size', 'N/A'),
                'Scaling': config.get('scaling', 'N/A')
            })
            
        except Exception as e:
            st.warning(f"Erro ao ler {result_file}: {str(e)}")
    
    if not models_data:
        st.warning("Nenhum dado de modelo válido encontrado.")
        return
    
    df_models = pd.DataFrame(models_data)
    
    st.write("**Comparação de Todos os Modelos:**")
    st.dataframe(df_models.round(4), use_container_width=True)
    
    render_model_comparison_charts(df_models)
    render_best_models_summary(df_models)

def render_model_comparison_charts(df_models):
    st.write("**Análises Visuais:**")
    
    tab1, tab2, tab3 = st.tabs(["Performance por Algoritmo", "Evolução Temporal", "Métricas Detalhadas"])
    
    with tab1:
        if len(df_models['Algoritmo'].unique()) > 1:
            metrics = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
            
            fig = px.box(df_models.melt(id_vars=['Algoritmo'], value_vars=metrics),
                        x='Algoritmo', y='value', color='variable',
                        title='Distribuição de Métricas por Algoritmo')
            fig.update_layout(yaxis_title='Score')
            st.plotly_chart(fig, use_container_width=True)
            
            avg_by_algorithm = df_models.groupby('Algoritmo')[metrics].mean().round(4)
            st.write("**Performance Média por Algoritmo:**")
            st.dataframe(avg_by_algorithm, use_container_width=True)
        else:
            st.info("Apenas um algoritmo encontrado. Execute modelos com diferentes algoritmos para comparação.")
    
    with tab2:
        if 'Timestamp' in df_models.columns:
            df_sorted = df_models.sort_values('Timestamp')
            
            fig = px.line(df_sorted, x=range(len(df_sorted)), y='Accuracy',
                         title='Evolução da Accuracy ao Longo do Tempo',
                         markers=True)
            fig.update_layout(xaxis_title='Ordem de Execução', yaxis_title='Accuracy')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.write("**Correlação entre Métricas:**")
        metrics_cols = ['Accuracy', 'F1-Score', 'Precision', 'Recall']
        corr_matrix = df_models[metrics_cols].corr()
        
        fig = px.imshow(corr_matrix, text_auto=True, aspect="auto",
                       title='Matriz de Correlação entre Métricas')
        st.plotly_chart(fig, use_container_width=True)

def render_best_models_summary(df_models):
    st.markdown("---")
    st.write("**Resumo dos Melhores Modelos:**")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        best_accuracy = df_models.loc[df_models['Accuracy'].idxmax()]
        st.metric("Melhor Accuracy", 
                 f"{best_accuracy['Accuracy']:.4f}",
                 f"{best_accuracy['Algoritmo']} - {best_accuracy['Dataset']}")
    
    with col2:
        best_f1 = df_models.loc[df_models['F1-Score'].idxmax()]
        st.metric("Melhor F1-Score", 
                 f"{best_f1['F1-Score']:.4f}",
                 f"{best_f1['Algoritmo']} - {best_f1['Dataset']}")
    
    with col3:
        avg_accuracy = df_models['Accuracy'].mean()
        st.metric("Accuracy Média Geral", f"{avg_accuracy:.4f}")
    
    st.write("**Top 5 Modelos por Accuracy:**")
    top_5 = df_models.nlargest(5, 'Accuracy')[['Modelo', 'Algoritmo', 'Dataset', 'Accuracy', 'F1-Score']]
    st.dataframe(top_5.round(4), use_container_width=True)