import streamlit as st
import os

def model_sidebar():
    st.sidebar.write("**Modelo Salvo Ativo**")
    
    models_dir = "data/models"
    
    if not os.path.exists(models_dir):
        st.sidebar.warning("Nenhum modelo salvo encontrado")
        st.sidebar.info("Execute um pipeline primeiro")
        return
    
    model_files = [f for f in os.listdir(models_dir) if f.endswith('.pkl')]
    
    if not model_files:
        st.sidebar.warning("Nenhum modelo salvo encontrado")
        st.sidebar.info("Execute um pipeline primeiro")
        return
    
    st.sidebar.success(f"Modelos disponíveis: {len(model_files)}")
    
    if 'loaded_model_name' in st.session_state:
        st.sidebar.info(f"Modelo carregado: {st.session_state.loaded_model_name}")
    else:
        st.sidebar.info("Nenhum modelo carregado")
    
    latest_model = max(model_files)
    st.sidebar.write(f"Mais recente: {latest_model}")
    
    if st.sidebar.button("Ir para Modelos", use_container_width=True):
        st.session_state.categoria = "Modelo"