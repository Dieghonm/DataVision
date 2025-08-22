import streamlit as st
from src.sidebar import render_sidebar
from src.main import render_main

st.set_page_config(
    page_title="DataVision EBAC SEMANTIX",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

def inicializar_session_state():
    defaults = {
        "categoria": "Pipeline",
        "data_source": None,
        "DF": None,
        "pipeline_config": None,
        "current_data": None,
        "dataset_name": None,
        "executions": []
    }
    
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

def main():
    st.title("DataVision EBAC SEMANTIX")
    st.markdown("---")
    inicializar_session_state()
    render_sidebar()
    render_main()

if __name__ == "__main__":
    main()