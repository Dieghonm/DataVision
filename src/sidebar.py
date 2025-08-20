import streamlit as st

from scripts.sidbar.modelo import model_sidebar
from scripts.sidbar.pipeline import pipeline_sidebar

def render_sidebar():
    categoria = st.sidebar.radio(
        "Escolha a Operação:",
        ["Pipeline de Dados", "Utilizar Modelo Salvo"]
    )

    st.sidebar.markdown("---")

    if categoria == "Pipeline de Dados":
        st.session_state.categoria = "Pipeline"
        pipeline_config = pipeline_sidebar()
        st.session_state.pipeline_config = pipeline_config
        
        if pipeline_config and pipeline_config.get('data_source'):
            st.sidebar.success(f"Dataset: {pipeline_config['data_source']}")

    else:
        st.session_state.categoria = "Modelo"
        model_sidebar()

