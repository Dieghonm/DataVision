import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import streamlit as st
import os
import io

def load_dataset(data_source, uploaded_file=None):
    """
    Carrega um dataset baseado na fonte especificada
    
    Args:
        data_source (str): Fonte do dataset
        uploaded_file: Arquivo enviado pelo usuário
    
    Returns:
        tuple: (DataFrame, nome_dataset) ou (None, None) em caso de erro
    """
    try:
        if data_source == "upload" and uploaded_file is not None:
            # Melhor tratamento para upload de arquivos
            try:
                # Ler o arquivo como string primeiro para detectar problemas
                if hasattr(uploaded_file, 'read'):
                    # Reset do ponteiro do arquivo
                    uploaded_file.seek(0)
                    
                    # Detectar encoding
                    sample = uploaded_file.read(10000)
                    uploaded_file.seek(0)
                    
                    # Tentar diferentes encodings
                    encodings_to_try = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
                    df = None
                    
                    for encoding in encodings_to_try:
                        try:
                            uploaded_file.seek(0)
                            df = pd.read_csv(uploaded_file, encoding=encoding)
                            st.info(f"Arquivo carregado com encoding: {encoding}")
                            break
                        except (UnicodeDecodeError, pd.errors.ParserError):
                            continue
                    
                    if df is None:
                        # Última tentativa com parâmetros mais flexíveis
                        uploaded_file.seek(0)
                        df = pd.read_csv(
                            uploaded_file, 
                            encoding='utf-8',
                            sep=None,  # Detectar separador automaticamente
                            engine='python',
                            encoding_errors='replace'
                        )
                    
                    # Validações do dataset
                    if df.empty:
                        st.error("O arquivo CSV está vazio.")
                        return None, None
                    
                    # Limpar nomes das colunas (remover espaços)
                    df.columns = df.columns.str.strip()
                    
                    # Se há uma coluna que pode ser target, criar target padrão
                    _ensure_target_column(df)
                    
                    return df, f"Dataset Personalizado ({uploaded_file.name})"
                
            except Exception as e:
                st.error(f"Erro ao processar arquivo CSV: {str(e)}")
                st.error("Verifique se o arquivo está no formato CSV correto.")
                return None, None

        elif data_source == "iris":
            data = load_iris()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            # Adicionar nomes das classes
            target_names = data.target_names
            df['target_name'] = df['target'].map({i: name for i, name in enumerate(target_names)})
            return df, "Iris Dataset"

        elif data_source == "wine":
            data = load_wine()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            # Adicionar nomes das classes
            target_names = data.target_names
            df['target_name'] = df['target'].map({i: f"Class_{name}" for i, name in enumerate(target_names)})
            return df, "Wine Dataset"

        elif data_source == "breast_cancer":
            data = load_breast_cancer()
            df = pd.DataFrame(data.data, columns=data.feature_names)
            df['target'] = data.target
            # Adicionar nomes das classes (0: malignant, 1: benign)
            df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})
            return df, "Breast Cancer Dataset"

        elif data_source == "Credit":
            file_path = "data/raw/credit_scoring.ftr"
            df = pd.read_feather(file_path)
            _ensure_target_column(df)
            return df, "Credit Scoring Dataset"

        elif data_source == "Hipertension":
            file_path = "data/raw/hypertension_dataset.csv"
            df = pd.read_csv(file_path)
            _ensure_target_column(df)
            return df, "Hypertension Dataset"

        elif data_source == "Phone addiction":
            file_path = "data/raw/teen_phone_addiction_dataset.csv"
            df = pd.read_csv(file_path)
            _ensure_target_column(df)
            return df, "Teen Phone Addiction Dataset"
        
        elif data_source == "upload":
            return None, None

        else:
            st.error(f"Fonte de dados '{data_source}' não reconhecida.")
            return None, None

    except FileNotFoundError as e:
        st.error(f"Arquivo não encontrado: {str(e)}")
        return None, None
    except pd.errors.EmptyDataError:
        st.error("O arquivo está vazio ou corrompido.")
        return None, None
    except pd.errors.ParserError as e:
        st.error(f"Erro ao analisar o arquivo: {str(e)}")
        st.error("Dica: Verifique se o separador do CSV está correto (vírgula, ponto-e-vírgula, tab)")
        return None, None
    except UnicodeDecodeError as e:
        st.error(f"Erro de encoding do arquivo: {str(e)}")
        st.error("Dica: Tente salvar o arquivo com encoding UTF-8")
        return None, None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar dataset: {str(e)}")
        st.error("Verifique se o arquivo está no formato correto.")
        return None, None


def _ensure_target_column(df):
    """
    Garante que o DataFrame tenha uma coluna target adequada
    """
    if 'target' not in df.columns:
        # Procurar por colunas que possam ser target
        possible_targets = ['Class', 'class', 'label', 'Label', 'y', 'Y', 
                          'outcome', 'result', 'diagnosis', 'approved', 
                          'default', 'risk', 'loan_status', 'hypertension',
                          'bp_status', 'addiction', 'addicted', 'phone_addiction',
                          'target_class', 'classification', 'category']
        
        for col in possible_targets:
            if col in df.columns:
                df['target'] = df[col]
                break
        
        # Se ainda não tem target, usar a última coluna
        if 'target' not in df.columns:
            last_col = df.columns[-1]
            # Verificar se a última coluna pode ser um target (poucos valores únicos)
            if df[last_col].nunique() <= 10:
                df['target'] = df[last_col]
    
    # Criar target_name se não existir
    if 'target_name' not in df.columns and 'target' in df.columns:
        unique_targets = df['target'].unique()
        
        # Se é numérico com poucos valores, criar mapeamento
        if df['target'].dtype in ['int64', 'float64'] and len(unique_targets) <= 10:
            if len(unique_targets) == 2:
                # Binário - tentar inferir o significado baseado no dataset
                min_val, max_val = sorted(unique_targets)
                
                # Mapeamentos específicos por contexto
                if any(col in df.columns for col in ['credit_score', 'income', 'loan']):
                    # Context: Credit scoring
                    df['target_name'] = df['target'].map({
                        min_val: 'Denied', 
                        max_val: 'Approved'
                    })
                elif any(col in df.columns for col in ['bp', 'blood_pressure', 'hypertension']):
                    # Context: Hypertension
                    df['target_name'] = df['target'].map({
                        min_val: 'Normal', 
                        max_val: 'Hypertensive'
                    })
                elif any(col in df.columns for col in ['phone', 'usage', 'addiction']):
                    # Context: Phone addiction
                    df['target_name'] = df['target'].map({
                        min_val: 'Normal', 
                        max_val: 'Addicted'
                    })
                else:
                    # Generic binary
                    df['target_name'] = df['target'].map({
                        min_val: f'Class_{min_val}', 
                        max_val: f'Class_{max_val}'
                    })
                    
            elif len(unique_targets) == 3:
                # 3 classes - verificar se é diabetes ou genérico
                sorted_targets = sorted(unique_targets)
                
                if any(col.lower() in ['glucose', 'hba1c', 'diabetes', 'blood_sugar'] for col in df.columns):
                    # Context: Diabetes
                    diabetes_mapping = {
                        sorted_targets[0]: 'Normal',
                        sorted_targets[1]: 'Pre-Diabetes', 
                        sorted_targets[2]: 'Diabetes'
                    }
                    df['target_name'] = df['target'].map(diabetes_mapping)
                else:
                    # Generic 3-class
                    df['target_name'] = df['target'].map({
                        val: f'Class_{val}' for val in sorted_targets
                    })
                    
            else:
                # Mais classes - usar números
                df['target_name'] = df['target'].astype(str)
        else:
            # Se já é string/object, usar como está
            df['target_name'] = df['target'].astype(str)


def validate_dataset(df):
    """
    Valida se um dataset está em formato adequado
    
    Args:
        df: DataFrame para validar
    
    Returns:
        tuple: (is_valid, error_message)
    """
    if df is None:
        return False, "Dataset é None"
    
    if df.empty:
        return False, "Dataset está vazio"
    
    if len(df.columns) < 2:
        return False, "Dataset deve ter pelo menos 2 colunas"
    
    # Verificar se há muitos valores faltantes
    missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
    if missing_ratio > 0.5:
        return False, "Dataset tem mais de 50% de valores faltantes"
    
    return True, "Dataset válido"


def preview_csv_content(uploaded_file):
    """
    Função para preview do conteúdo do CSV antes do carregamento completo
    """
    if uploaded_file is not None:
        try:
            uploaded_file.seek(0)
            # Ler apenas as primeiras linhas para preview
            sample_lines = []
            for i, line in enumerate(uploaded_file):
                if i >= 5:  # Apenas 5 linhas
                    break
                if isinstance(line, bytes):
                    line = line.decode('utf-8', errors='replace')
                sample_lines.append(line.strip())
            
            uploaded_file.seek(0)  # Reset para uso posterior
            return sample_lines
            
        except Exception as e:
            return [f"Erro ao ler preview: {str(e)}"]
    return []


def check_data_files():
    """
    Verifica quais arquivos de dados estão disponíveis na pasta data/raw/
    
    Returns:
        dict: Status de cada arquivo esperado
    """
    expected_files = {
        'credit_scoring.ftr': 'Credit Scoring Dataset',
        'hypertension_dataset.csv': 'Hypertension Dataset', 
        'teen_phone_addiction_dataset.csv': 'Teen Phone Addiction Dataset'
    }
    
    status = {}
    
    for filename, description in expected_files.items():
        filepath = f"data/raw/{filename}"
        
        if os.path.exists(filepath):
            if filename.endswith('.ftr'):
                df = pd.read_feather(filepath)
            else:
                df = pd.read_csv(filepath, nrows=5)
            
            status[filename] = {
                'exists': True,
                'readable': True,
                'description': description,
                'rows': len(df) if filename.endswith('.ftr') else 'N/A (CSV)',
                'columns': len(df.columns),
                'size_mb': round(os.path.getsize(filepath) / (1024*1024), 2)
            }
        else:
            status[filename] = {
                'exists': False,
                'readable': False,
                'description': description
            }
    
    return status


def display_data_status():
    """
    Exibe o status dos arquivos de dados na interface do Streamlit
    """
    st.subheader("Status dos Datasets")
    
    status = check_data_files()
    
    for filename, info in status.items():
        with st.expander(f"{info['description']} ({filename})"):
            if info['exists'] and info['readable']:
                st.success("Arquivo disponível e legível")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    if 'rows' in info and info['rows'] != 'N/A (CSV)':
                        st.metric("Linhas", info['rows'])
                with col2:
                    st.metric("Colunas", info['columns'])
                with col3:
                    st.metric("Tamanho", f"{info['size_mb']} MB")
                    
            else:
                st.warning("Arquivo não encontrado")
                st.info(f"Coloque o arquivo em: data/raw/{filename}")