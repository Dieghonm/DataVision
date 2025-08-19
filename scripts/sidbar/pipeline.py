import streamlit as st
from zold.data_loader import load_dataset

class PipelineUI:
    def __init__(self):
        self.current_data = None
        self._init_session_state()

    def _init_session_state(self):
        """Inicializa o estado da sessão"""
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'dataset_name' not in st.session_state:
            st.session_state.dataset_name = None
        if 'last_data_source' not in st.session_state:
            st.session_state.last_data_source = None
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None

    def _load_dataset(self, data_source, uploaded_file=None):
        """Carrega um dataset"""
        try:
            df, name = load_dataset(data_source, uploaded_file)
            return df, name
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {str(e)}")
            return None, None

    def _render_algorithm_params(self, algorithm):
        """Renderiza parâmetros específicos do algoritmo"""
        st.sidebar.markdown("---")
        st.sidebar.subheader("🔧 Parâmetros do Algoritmo")
        
        if algorithm == "random_forest":
            n_estimators = st.sidebar.slider(
                "N Estimators", 
                min_value=10, max_value=500, value=100, step=10,
                help="Número de árvores na floresta"
            )
            max_depth = st.sidebar.slider(
                "Max Depth", 
                min_value=1, max_value=50, value=10,
                help="Profundidade máxima das árvores"
            )
            return {"n_estimators": n_estimators, "max_depth": max_depth}

        elif algorithm == "logistic_regression":
            c_value = st.sidebar.slider(
                "C (Regularização)", 
                min_value=0.01, max_value=10.0, value=1.0, step=0.01,
                help="Força da regularização (valores menores = mais regularização)"
            )
            max_iter = st.sidebar.slider(
                "Max Iterations", 
                min_value=100, max_value=1000, value=100, step=50,
                help="Número máximo de iterações"
            )
            return {"C": c_value, "max_iter": max_iter}

        elif algorithm == "svm":
            kernel = st.sidebar.selectbox(
                "Kernel", 
                ["linear", "rbf", "poly"],
                help="Tipo de kernel para o SVM"
            )
            c_value = st.sidebar.slider(
                "C (Regularização)", 
                min_value=0.01, max_value=10.0, value=1.0, step=0.01,
                help="Parâmetro de regularização"
            )
            return {"kernel": kernel, "C": c_value}

        return {}

    def _execute_pipeline(self, config):
        """Executa o pipeline com as configurações fornecidas"""
        try:
            st.success("🚀 Pipeline iniciado!")
            
            # Criar uma barra de progresso
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Simular etapas do pipeline
            steps = [
                "Carregando dados...",
                "Preprocessando dados...",
                "Dividindo dataset...",
                "Treinando modelo...",
                "Avaliando modelo...",
                "Salvando resultados..."
            ]
            
            for i, step in enumerate(steps):
                status_text.text(step)
                progress_bar.progress((i + 1) / len(steps))
                # Simular tempo de processamento
                import time
                time.sleep(0.5)
            
            status_text.text("✅ Pipeline concluído!")
            
            # Mostrar configuração utilizada
            with st.expander("📋 Configuração Utilizada"):
                col1, col2 = st.columns(2)
                with col1:
                    st.json({
                        "data_source": config["data_source"],
                        "algorithm": config["algorithm"],
                        "test_size": config["test_size"],
                        "cv_folds": config["cv_folds"]
                    })
                with col2:
                    st.json({
                        "metrics": config["metrics"],
                        "algo_params": config["algo_params"],
                        "steps": config["steps"]
                    })
            
            st.info("⚠️ Esta é uma simulação. Conecte as funções reais do pipeline aqui.")
            
        except Exception as e:
            st.error(f"Erro na execução do pipeline: {str(e)}")

    def render_pipeline(self):
        """Renderiza a sidebar do pipeline UI"""
        st.sidebar.title("⚙️ Configurações do Pipeline")

        # 📂 Fonte dos dados
        st.sidebar.subheader("📂 Dados")
        data_source = st.sidebar.selectbox(
            "Fonte de dados:",
            ["upload", "iris", "wine", "breast_cancer", "Credit", "Hipertension", "Phone addiction"],
            help="Escolha um dataset pré-definido ou faça upload",
            key="data_source_select"
        )

        uploaded_file = None
        if data_source == "upload":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV:", 
                type=['csv'],
                help="Faça upload do seu dataset em CSV",
                key="file_upload"
            )

        # Verificar se precisa recarregar os dados
        data_changed = (
            data_source != st.session_state.get('last_data_source') or
            uploaded_file != st.session_state.get('last_uploaded_file')
        )

        if data_changed:
            with st.spinner("Carregando dataset..."):
                self.current_data, dataset_name = self._load_dataset(data_source, uploaded_file)
                st.session_state.current_data = self.current_data
                st.session_state.dataset_name = dataset_name
                st.session_state.last_data_source = data_source
                st.session_state.last_uploaded_file = uploaded_file

        # Recuperar dados da sessão
        self.current_data = st.session_state.get('current_data')

        # Mostrar info do dataset se carregado
        if self.current_data is not None:
            st.sidebar.success(f"✅ {st.session_state.get('dataset_name', 'Dataset')} carregado")
            st.sidebar.info(f"📊 Shape: {self.current_data.shape}")

        # 🤖 Modelo
        st.sidebar.subheader("🤖 Modelo")
        algorithm = st.sidebar.selectbox(
            "Algoritmo:",
            ["random_forest", "logistic_regression", "svm"],
            help="Escolha o algoritmo de ML"
        )

        algo_params = self._render_algorithm_params(algorithm)

        # 📊 Avaliação
        st.sidebar.subheader("📊 Avaliação")
        test_size = st.sidebar.slider(
            "Tamanho do teste:", 
            min_value=0.1, max_value=0.5, value=0.2, step=0.05,
            help="Proporção dos dados para teste"
        )
        cv_folds = st.sidebar.slider(
            "Cross-validation:", 
            min_value=3, max_value=10, value=5,
            help="Número de folds para validação cruzada"
        )

        metrics = st.sidebar.multiselect(
            "Métricas:",
            ["accuracy", "precision", "recall", "f1", "roc_auc"],
            default=["accuracy", "f1"],
            help="Métricas de avaliação do modelo"
        )

        # 🔧 Pipeline
        st.sidebar.subheader("🔧 Pipeline")
        steps = st.sidebar.multiselect(
            "Etapas:",
            ["load_data", "preprocess_data", "train_model", "evaluate_model", "save_results"],
            default=["load_data", "preprocess_data", "train_model", "evaluate_model"],
            help="Etapas do pipeline a serem executadas"
        )

        # ⚙️ Configurações Avançadas
        with st.sidebar.expander("⚙️ Configurações Avançadas"):
            scaling = st.selectbox(
                "Normalização:", 
                ["standard", "minmax", "robust", "none"],
                help="Tipo de normalização dos dados"
            )
            random_state = st.number_input(
                "Random State:", 
                value=42, min_value=0, max_value=999,
                help="Semente para reprodutibilidade"
            )

        # 🚀 Executar Pipeline
        st.sidebar.markdown("---")
        if self.current_data is not None:
            if st.sidebar.button("🚀 Executar Pipeline", type="primary", use_container_width=True):
                config = {
                    "data_source": data_source,
                    "uploaded_file": uploaded_file,
                    "algorithm": algorithm,
                    "algo_params": algo_params,
                    "test_size": test_size,
                    "cv_folds": cv_folds,
                    "metrics": metrics,
                    "steps": steps,
                    "scaling": scaling,
                    "random_state": random_state
                }
                self._execute_pipeline(config)
        else:
            st.sidebar.warning("👆 Selecione um dataset para continuar")

        return {
            "data_source": data_source,
            "uploaded_file": uploaded_file,
            "algorithm": algorithm,
            "algo_params": algo_params,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "metrics": metrics,
            "steps": steps,
            "scaling": scaling,
            "random_state": random_state
        }

def pipeline_sidebar():
    """Função principal para renderizar o sidebar do pipeline"""
    pipeline_ui = PipelineUI()
    return pipeline_ui.render_pipeline()