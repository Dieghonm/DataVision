# from scripts.sidebar.modelo import model_sidebar
from scripts.sidebar.pipeline import pipeline_sidebar
import streamlit as st

def render_sidebar():
    st.sidebar.header("🎛️ Painel de Controle")
    categoria = st.sidebar.radio(
        "⚙️ Escolha a Operação:",
        ["Pipeline de Dados", "Utilizar Modelo Salvo"]
    )

    if categoria == "Pipeline de Dados":
        st.session_state.categoria = "Pipeline"
        pipeline_sidebar()
    else:
        st.session_state.categoria = "Modelo"
        # model_sidebar()

