import streamlit as st
import pandas as pd
import numpy as np

def model_page(params):
    """Página para utilização de modelos salvos"""
    st.subheader("🤖 Utilização de Modelos Salvos")

    modelo = params.get("modelo")
    arquivo_pred = params.get("arquivo_pred")

    # Informações sobre o modelo selecionado
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.write(f"**Modelo selecionado:** {modelo}")
        
        # Informações fictícias sobre os modelos
        model_info = {
            "Modelo_1.pkl": {
                "algoritmo": "Random Forest",
                "acurácia": "94.2%",
                "dataset_treino": "Iris Dataset",
                "data_treino": "2024-01-15"
            },
            "Modelo_2.pkl": {
                "algoritmo": "Logistic Regression", 
                "acurácia": "87.5%",
                "dataset_treino": "Credit Scoring",
                "data_treino": "2024-01-10"
            },
            "Modelo_3.pkl": {
                "algoritmo": "SVM",
                "acurácia": "91.8%", 
                "dataset_treino": "Breast Cancer",
                "data_treino": "2024-01-20"
            }
        }
        
        if modelo in model_info:
            info = model_info[modelo]
            with st.expander("ℹ️ Informações do Modelo"):
                col_a, col_b = st.columns(2)
                with col_a:
                    st.metric("Algoritmo", info["algoritmo"])
                    st.metric("Dataset de Treino", info["dataset_treino"])
                with col_b:
                    st.metric("Acurácia", info["acurácia"])
                    st.metric("Data de Treino", info["data_treino"])

    with col2:
        st.info("💡 **Dica:** Certifique-se de que seus dados tenham as mesmas colunas usadas no treinamento!")

    # Upload e processamento do arquivo
    if arquivo_pred is not None:
        try:
            # Carregar arquivo
            with st.spinner("Carregando arquivo..."):
                if arquivo_pred.name.endswith(".csv"):
                    df_pred = pd.read_csv(arquivo_pred)
                else:
                    df_pred = pd.read_excel(arquivo_pred)
            
            with col1:
                st.write("### 📊 Dados enviados para predição:")
                st.dataframe(df_pred.head(10))
                
            with col2:
                st.write("### 📈 Resumo dos Dados:")
                st.metric("Linhas", df_pred.shape[0])
                st.metric("Colunas", df_pred.shape[1])
                
                # Verificar tipos de dados
                numeric_cols = df_pred.select_dtypes(include=[np.number]).columns.tolist()
                categorical_cols = df_pred.select_dtypes(include=['object']).columns.tolist()
                
                st.write(f"**Numéricas:** {len(numeric_cols)}")
                st.write(f"**Categóricas:** {len(categorical_cols)}")

            # Análise básica dos dados
            if st.checkbox("🔍 Mostrar análise exploratória"):
                tab1, tab2, tab3 = st.tabs(["Estatísticas", "Valores Faltantes", "Tipos de Dados"])
                
                with tab1:
                    if numeric_cols:
                        st.write("**Estatísticas das variáveis numéricas:**")
                        st.dataframe(df_pred[numeric_cols].describe())
                    else:
                        st.info("Nenhuma coluna numérica encontrada.")
                
                with tab2:
                    missing_data = df_pred.isnull().sum()
                    if missing_data.any():
                        st.write("**Valores faltantes por coluna:**")
                        missing_df = pd.DataFrame({
                            'Coluna': missing_data.index,
                            'Valores Faltantes': missing_data.values,
                            'Percentual': (missing_data.values / len(df_pred)) * 100
                        })
                        missing_df = missing_df[missing_df['Valores Faltantes'] > 0]
                        st.dataframe(missing_df)
                    else:
                        st.success("✅ Nenhum valor faltante encontrado!")
                
                with tab3:
                    types_df = pd.DataFrame({
                        'Coluna': df_pred.columns,
                        'Tipo': df_pred.dtypes.astype(str)
                    })
                    st.dataframe(types_df)

            # Botão para fazer predições
            st.markdown("---")
            if st.button("🔮 Fazer Predições", type="primary", use_container_width=True):
                with st.spinner("Processando predições..."):
                    # Simular predições
                    import time
                    import random
                    time.sleep(2)
                    
                    # Gerar predições fictícias
                    n_samples = len(df_pred)
                    if "iris" in modelo.lower() or "Modelo_1" in modelo:
                        predictions = np.random.choice(['Setosa', 'Versicolor', 'Virginica'], n_samples)
                        probabilities = np.random.random((n_samples, 3))
                        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                    else:
                        predictions = np.random.choice([0, 1], n_samples)
                        probabilities = np.random.random((n_samples, 2))
                        probabilities = probabilities / probabilities.sum(axis=1, keepdims=True)
                    
                    # Mostrar resultados
                    st.success("✅ Predições concluídas!")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("### 📊 Resultados das Predições:")
                        results_df = df_pred.copy()
                        results_df['Predição'] = predictions
                        results_df['Confiança'] = np.max(probabilities, axis=1)
                        st.dataframe(results_df[['Predição', 'Confiança']].head(10))
                    
                    with col2:
                        st.write("### 📈 Distribuição das Predições:")
                        pred_counts = pd.Series(predictions).value_counts()
                        st.bar_chart(pred_counts)
                    
                    # Opção para download
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download dos Resultados",
                        data=csv,
                        file_name=f"predicoes_{modelo.replace('.pkl', '')}.csv",
                        mime="text/csv"
                    )

            st.warning("⚠️ **Nota:** Esta é uma simulação. Em produção, aqui seria carregado o modelo real e aplicadas as predições.")
            
        except Exception as e:
            st.error(f"❌ Erro ao processar o arquivo: {str(e)}")
            st.info("💡 Verifique se o arquivo está no formato correto (CSV ou Excel) e não está corrompido.")
    
    else:
        # Tela de instruções quando não há arquivo
        st.info("📁 Envie um arquivo para o modelo fazer predições.")
        
        st.markdown("""
        ### 📋 Instruções de Uso:
        
        1. **Selecione um modelo** na barra lateral
        2. **Faça upload** de um arquivo CSV ou Excel com os dados
        3. **Verifique** se as colunas correspondem ao modelo escolhido
        4. **Clique em "Fazer Predições"** para obter os resultados
        5. **Baixe** os resultados em CSV
        
        ### ⚠️ Requisitos dos Dados:
        - Formato: CSV ou Excel (.xlsx)
        - Colunas devem corresponder às features do modelo
        - Dados limpos e pré-processados
        - Sem valores faltantes nas colunas essenciais
        """)
        
        # Exemplo de formato esperado
        with st.expander("📝 Exemplo de Formato de Dados"):
            if modelo == "Modelo_1.pkl":  # Iris model
                example_data = {
                    'sepal_length': [5.1, 4.9, 4.7],
                    'sepal_width': [3.5, 3.0, 3.2], 
                    'petal_length': [1.4, 1.4, 1.3],
                    'petal_width': [0.2, 0.2, 0.2]
                }
            elif modelo == "Modelo_2.pkl":  # Credit model
                example_data = {
                    'income': [50000, 30000, 75000],
                    'age': [35, 25, 45],
                    'credit_score': [700, 650, 800]
                }
            else:  # Breast cancer model
                example_data = {
                    'mean_radius': [17.99, 20.57, 19.69],
                    'mean_texture': [10.38, 17.77, 21.25],
                    'mean_perimeter': [122.8, 132.9, 130.0]
                }
            
            example_df = pd.DataFrame(example_data)
            st.dataframe(example_df)
            st.caption("⬆️ Exemplo de formato esperado para este modelo")
