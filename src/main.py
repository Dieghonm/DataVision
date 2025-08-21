import streamlit as st

from scripts.mainPage.modelo import model_main
from scripts.mainPage.pipeline import pipeline_main
from scripts.mainPage.model_run import model_run

def render_main():
    categoria = st.session_state.categoria

    if categoria == "Pipeline":
        pipeline_main()
    if categoria == "Model_Run":
        model_run()
    elif categoria == "Modelo":
        model_main()
    else:
        st.error("Categoria não reconhecida. Por favor, selecione uma opção válida.")