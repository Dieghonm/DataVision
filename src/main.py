import streamlit as st

from scripts.mainPage.modelo import model_main
from scripts.mainPage.pipeline import pipeline_main
from scripts.mainPage.model_run import model_run

def render_main():
    """
    Renderiza a página principal baseada na categoria selecionada
    """
    categoria = st.session_state.get('categoria', 'Pipeline')

    if categoria == "Pipeline":
        pipeline_main()
    elif categoria == "Model_Run":
        if ('pipeline_config' not in st.session_state or 
            st.session_state.pipeline_config is None):
            st.warning("⚠️ Nenhuma configuração de pipeline encontrada!")
            st.info("Configure o pipeline na seção 'Pipeline de Dados' primeiro.")
            
            if st.button("🔙 Voltar ao Pipeline"):
                st.session_state.categoria = "Pipeline"
                st.rerun()
        else:
            model_run()
    elif categoria == "Modelo":
        model_main()
    else:
        st.error("Categoria não reconhecida. Por favor, selecione uma opção válida.")
        
        if st.button("🏠 Voltar ao Início"):
            st.session_state.categoria = "Pipeline"
            st.rerun()