import streamlit as st
from src.sidebar import render_sidebar

st.set_page_config(
    page_title="DataVision EBAC SEMANTIX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inicialização do estado da sessão
def inicializar_session_state():
    if "categoria" not in st.session_state:
        st.session_state.categoria = "Pipeline"

def main():
    st.title("DataVision EBAC SEMANTIX")
    st.markdown("---")
    inicializar_session_state()
    render_sidebar()

if __name__ == "__main__":
    main()


