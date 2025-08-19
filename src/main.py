import streamlit as st

from scripts.mainPage.modelo import model_main
from scripts.mainPage.pipeline import pipeline_main

def render_main():
    categoria = st.session_state.categoria

    if categoria == "Pipeline de Dados":
        st.session_state.categoria = "Pipeline"
        pipeline_main()
    else:
        st.session_state.categoria = "Modelo"
        model_main()