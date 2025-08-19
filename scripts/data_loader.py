import pandas as pd
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import streamlit as st
import os

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
            df = pd.read_csv(uploaded_file)
            return df, f"Dataset Personalizado ({uploaded_file.name})"

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
            # Tentar carregar o arquivo real primeiro
            file_path = "data/raw/credit_scoring.ftr"
            if os.path.exists(file_path):
                try:
                    df = pd.read_feather(file_path)
                    # Garantir que existe uma coluna target
                    if 'target' not in df.columns:
                        # Procurar por colunas que possam ser target
                        possible_targets = ['approved', 'default', 'risk', 'loan_status', 'result']
                        for col in possible_targets:
                            if col in df.columns:
                                df['target'] = df[col]
                                break
                    
                    # Criar target_name se não existir
                    if 'target_name' not in df.columns and 'target' in df.columns:
                        unique_targets = df['target'].unique()
                        if len(unique_targets) == 2:
                            # Binário - assumir 0=Negado, 1=Aprovado
                            df['target_name'] = df['target'].map({0: 'Denied', 1: 'Approved'})
                        else:
                            df['target_name'] = df['target'].astype(str)
                    
                    return df, "Credit Scoring Dataset"
                except Exception as e:
                    st.warning(f"⚠️ Erro ao ler arquivo {file_path}: {str(e)}. Usando dados sintéticos.")
            else:
                st.warning(f"⚠️ Arquivo {file_path} não encontrado. Usando dados sintéticos.")
            
            # Se não conseguiu carregar, gerar dados sintéticos
            df = _generate_credit_data()
            return df, "Credit Scoring Dataset (Sintético)"

        elif data_source == "Hipertension":
            file_path = "data/raw/hypertension_dataset.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Garantir que existe uma coluna target
                    if 'target' not in df.columns:
                        # Procurar por colunas que possam ser target
                        possible_targets = ['hypertension', 'bp_status', 'diagnosis', 'result', 'class']
                        for col in possible_targets:
                            if col in df.columns:
                                df['target'] = df[col]
                                break
                    
                    # Criar target_name se não existir
                    if 'target_name' not in df.columns and 'target' in df.columns:
                        unique_targets = df['target'].unique()
                        if len(unique_targets) == 2:
                            # Binário - assumir 0=Normal, 1=Hipertensivo
                            df['target_name'] = df['target'].map({0: 'Normal', 1: 'Hypertensive'})
                        else:
                            df['target_name'] = df['target'].astype(str)
                    
                    return df, "Hypertension Dataset"
                except Exception as e:
                    st.warning(f"⚠️ Erro ao ler arquivo {file_path}: {str(e)}. Usando dados sintéticos.")
            else:
                st.warning(f"⚠️ Arquivo {file_path} não encontrado. Usando dados sintéticos.")
            
            # Se não conseguiu carregar, gerar dados sintéticos
            df = _generate_hypertension_data()
            return df, "Hypertension Dataset (Sintético)"

        elif data_source == "Phone addiction":
            file_path = "data/raw/teen_phone_addiction_dataset.csv"
            if os.path.exists(file_path):
                try:
                    df = pd.read_csv(file_path)
                    
                    # Garantir que existe uma coluna target
                    if 'target' not in df.columns:
                        # Procurar por colunas que possam ser target
                        possible_targets = ['addiction', 'addicted', 'phone_addiction', 'result', 'class', 'status']
                        for col in possible_targets:
                            if col in df.columns:
                                df['target'] = df[col]
                                break
                    
                    # Criar target_name se não existir
                    if 'target_name' not in df.columns and 'target' in df.columns:
                        unique_targets = df['target'].unique()
                        if len(unique_targets) == 2:
                            # Binário - assumir 0=Normal, 1=Viciado
                            df['target_name'] = df['target'].map({0: 'Normal', 1: 'Addicted'})
                        else:
                            df['target_name'] = df['target'].astype(str)
                    
                    return df, "Teen Phone Addiction Dataset"
                except Exception as e:
                    st.warning(f"⚠️ Erro ao ler arquivo {file_path}: {str(e)}. Usando dados sintéticos.")
            else:
                st.warning(f"⚠️ Arquivo {file_path} não encontrado. Usando dados sintéticos.")
            
            # Se não conseguiu carregar, gerar dados sintéticos
            df = _generate_phone_addiction_data()
            return df, "Teen Phone Addiction Dataset (Sintético)"
        
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
        return None, None
    except Exception as e:
        st.error(f"Erro inesperado ao carregar dataset: {str(e)}")
        return None, None


def _generate_credit_data():
    """Gera dados sintéticos para credit scoring"""
    import numpy as np
    
    n_samples = 1000
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(18, 80, n_samples),
        'income': np.random.normal(50000, 20000, n_samples),
        'credit_score': np.random.randint(300, 850, n_samples),
        'debt_to_income': np.random.uniform(0, 1, n_samples),
        'employment_length': np.random.randint(0, 30, n_samples),
        'loan_amount': np.random.normal(15000, 10000, n_samples),
    }
    
    # Target baseado em lógica simplificada
    target = []
    for i in range(n_samples):
        score = (
            (data['credit_score'][i] / 850) * 0.4 +
            (min(data['income'][i], 100000) / 100000) * 0.3 +
            (1 - data['debt_to_income'][i]) * 0.2 +
            (min(data['employment_length'][i], 10) / 10) * 0.1
        )
        target.append(1 if score > 0.6 else 0)
    
    data['target'] = target
    data['target_name'] = ['Approved' if t else 'Denied' for t in target]
    
    return pd.DataFrame(data)


def _generate_hypertension_data():
    """Gera dados sintéticos para hipertensão"""
    import numpy as np
    
    n_samples = 800
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(20, 80, n_samples),
        'gender': np.random.choice([0, 1], n_samples),  # 0: F, 1: M
        'bmi': np.random.normal(26, 5, n_samples),
        'smoking': np.random.choice([0, 1], n_samples),
        'alcohol': np.random.choice([0, 1], n_samples),
        'exercise': np.random.choice([0, 1], n_samples),
        'salt_intake': np.random.uniform(1, 10, n_samples),
        'family_history': np.random.choice([0, 1], n_samples),
        'systolic_bp': np.random.normal(125, 20, n_samples),
        'diastolic_bp': np.random.normal(80, 15, n_samples),
    }
    
    # Target baseado em fatores de risco
    target = []
    for i in range(n_samples):
        risk_score = (
            (data['age'][i] > 50) * 0.2 +
            (data['bmi'][i] > 30) * 0.2 +
            data['smoking'][i] * 0.15 +
            data['family_history'][i] * 0.15 +
            (data['salt_intake'][i] > 7) * 0.1 +
            (not data['exercise'][i]) * 0.1 +
            (data['systolic_bp'][i] > 140 or data['diastolic_bp'][i] > 90) * 0.3
        )
        target.append(1 if risk_score > 0.5 else 0)
    
    data['target'] = target
    data['target_name'] = ['Hypertensive' if t else 'Normal' for t in target]
    
    return pd.DataFrame(data)


def _generate_phone_addiction_data():
    """Gera dados sintéticos para vício em telefone"""
    import numpy as np
    
    n_samples = 600
    np.random.seed(42)
    
    data = {
        'age': np.random.randint(13, 19, n_samples),
        'gender': np.random.choice([0, 1], n_samples),
        'daily_usage_hours': np.random.gamma(3, 2, n_samples),  # Horas por dia
        'social_media_apps': np.random.randint(1, 15, n_samples),
        'notifications_per_day': np.random.poisson(50, n_samples),
        'sleep_hours': np.random.normal(7, 1.5, n_samples),
        'academic_performance': np.random.randint(1, 5, n_samples),  # 1-5 scale
        'social_interaction_score': np.random.randint(1, 10, n_samples),
        'anxiety_level': np.random.randint(1, 10, n_samples),
        'physical_activity_hours': np.random.gamma(1, 2, n_samples),
    }
    
    # Target baseado em padrões de uso problemático
    target = []
    for i in range(n_samples):
        addiction_score = (
            (data['daily_usage_hours'][i] > 6) * 0.25 +
            (data['notifications_per_day'][i] > 100) * 0.2 +
            (data['sleep_hours'][i] < 6) * 0.15 +
            (data['academic_performance'][i] < 3) * 0.15 +
            (data['social_interaction_score'][i] < 5) * 0.1 +
            (data['anxiety_level'][i] > 7) * 0.1 +
            (data['physical_activity_hours'][i] < 1) * 0.05
        )
        target.append(1 if addiction_score > 0.6 else 0)
    
    data['target'] = target
    data['target_name'] = ['Addicted' if t else 'Normal' for t in target]
    
    return pd.DataFrame(data)


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