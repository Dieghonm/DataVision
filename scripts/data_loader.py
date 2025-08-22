import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import streamlit as st
import os
from datetime import datetime

class DataLoader:
    def __init__(self):
        self.encodings = ['utf-8', 'latin-1', 'cp1252', 'iso-8859-1']
    
    def load_dataset(self, data_source, uploaded_file=None):
        try:
            if data_source == "upload" and uploaded_file is not None:
                return self._load_uploaded_file(uploaded_file)
            elif data_source in ["iris", "wine", "breast_cancer"]:
                return self._load_sklearn_dataset(data_source)
            elif data_source in ["Credit", "Hipertension", "Phone addiction"]:
                return self._load_local_dataset(data_source)
            else:
                return None, None
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {str(e)}")
            return None, None

    def _load_uploaded_file(self, uploaded_file):
        for encoding in self.encodings:
            try:
                uploaded_file.seek(0)
                df = pd.read_csv(uploaded_file, encoding=encoding)
                if not df.empty:
                    df.columns = df.columns.str.strip()
                    # CORREÇÃO: Tratar colunas datetime e limpar dados
                    df = self._clean_dataset(df)
                    self._ensure_target_column(df)
                    return df, f"Dataset Personalizado ({uploaded_file.name})"
            except (UnicodeDecodeError, pd.errors.ParserError):
                continue
        
        uploaded_file.seek(0)
        df = pd.read_csv(
            uploaded_file, 
            encoding='utf-8',
            sep=None,
            engine='python',
            encoding_errors='replace'
        )
        
        if df.empty:
            raise ValueError("O arquivo CSV está vazio.")
        
        df.columns = df.columns.str.strip()
        # CORREÇÃO: Tratar colunas datetime e limpar dados
        df = self._clean_dataset(df)
        self._ensure_target_column(df)
        return df, f"Dataset Personalizado ({uploaded_file.name})"

    def _clean_dataset(self, df):
        """
        NOVA FUNÇÃO: Limpa o dataset tratando diferentes tipos de dados problemáticos
        """
        df_cleaned = df.copy()
        
        # 1. Identificar e tratar colunas datetime
        datetime_columns = []
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Tentar converter para datetime
                try:
                    # Detectar possíveis formatos de data
                    sample_values = df_cleaned[col].dropna().head(10)
                    if len(sample_values) > 0:
                        # Verificar se parece com data
                        if self._looks_like_datetime(sample_values):
                            df_cleaned[col] = pd.to_datetime(df_cleaned[col], errors='coerce')
                            if df_cleaned[col].dtype == 'datetime64[ns]':
                                datetime_columns.append(col)
                except:
                    pass
        
        # 2. Tratar colunas datetime convertendo em features numéricas
        for col in datetime_columns:
            if col not in ['target', 'target_name']:  # Não processar se for target
                try:
                    # Extrair features de data
                    df_cleaned[f'{col}_year'] = df_cleaned[col].dt.year
                    df_cleaned[f'{col}_month'] = df_cleaned[col].dt.month
                    df_cleaned[f'{col}_day'] = df_cleaned[col].dt.day
                    df_cleaned[f'{col}_dayofweek'] = df_cleaned[col].dt.dayofweek
                    
                    # Se tem hora, extrair também
                    if df_cleaned[col].dt.hour.nunique() > 1:
                        df_cleaned[f'{col}_hour'] = df_cleaned[col].dt.hour
                    
                    # Remover coluna datetime original
                    df_cleaned = df_cleaned.drop(columns=[col])
                    st.info(f"Coluna datetime '{col}' convertida em features numéricas")
                except Exception as e:
                    st.warning(f"Erro ao processar coluna datetime '{col}': {str(e)}")
                    # Se não conseguir processar, remove a coluna
                    df_cleaned = df_cleaned.drop(columns=[col])
        
        # 3. Remover colunas com tipos problemáticos que não podem ser processados
        problematic_columns = []
        for col in df_cleaned.columns:
            if df_cleaned[col].dtype == 'object':
                # Verificar se é uma coluna de texto livre (muitos valores únicos)
                unique_ratio = df_cleaned[col].nunique() / len(df_cleaned)
                if unique_ratio > 0.8 and df_cleaned[col].nunique() > 50:
                    # Provavelmente é uma coluna de texto livre (IDs, comentários, etc)
                    problematic_columns.append(col)
                    
        # Remove colunas problemáticas (exceto se for target)
        for col in problematic_columns:
            if col not in ['target', 'target_name']:
                df_cleaned = df_cleaned.drop(columns=[col])
                st.warning(f"Coluna '{col}' removida (alta cardinalidade de texto)")
        
        # 4. Tratar valores infinitos
        numeric_cols = df_cleaned.select_dtypes(include=[np.number]).columns
        df_cleaned[numeric_cols] = df_cleaned[numeric_cols].replace([np.inf, -np.inf], np.nan)
        
        return df_cleaned
    
    def _looks_like_datetime(self, sample_values):
        """
        Verifica se os valores parecem ser datetime
        """
        datetime_patterns = [
            r'\d{4}-\d{2}-\d{2}',  # YYYY-MM-DD
            r'\d{2}/\d{2}/\d{4}',  # MM/DD/YYYY
            r'\d{2}-\d{2}-\d{4}',  # MM-DD-YYYY
            r'\d{4}/\d{2}/\d{2}',  # YYYY/MM/DD
        ]
        
        import re
        for value in sample_values.astype(str):
            for pattern in datetime_patterns:
                if re.search(pattern, str(value)):
                    return True
        return False

    def _load_sklearn_dataset(self, data_source):
        dataset_map = {
            "iris": (load_iris, "Iris Dataset"),
            "wine": (load_wine, "Wine Dataset"),
            "breast_cancer": (load_breast_cancer, "Breast Cancer Dataset")
        }
        
        loader, name = dataset_map[data_source]
        data = loader()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        
        if data_source == "breast_cancer":
            df['target_name'] = df['target'].map({0: 'malignant', 1: 'benign'})
        else:
            target_names = data.target_names
            df['target_name'] = df['target'].map({i: name for i, name in enumerate(target_names)})
        
        return df, name

    def _load_local_dataset(self, data_source):
        file_map = {
            "Credit": ("data/raw/credit_scoring.ftr", "Credit Scoring Dataset"),
            "Hipertension": ("data/raw/hypertension_dataset.csv", "Hypertension Dataset"),
            "Phone addiction": ("data/raw/teen_phone_addiction_dataset.csv", "Teen Phone Addiction Dataset")
        }
        
        file_path, name = file_map[data_source]
        
        if file_path.endswith('.ftr'):
            df = pd.read_feather(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # CORREÇÃO: Limpar dataset antes de processar target
        df = self._clean_dataset(df)
        self._ensure_target_column(df)
        return df, name

    def _ensure_target_column(self, df):
        if 'target' in df.columns:
            # CORREÇÃO: Verificar se target é contínuo e converter se necessário
            self._fix_target_column(df)
            return
        
        possible_targets = [
            'Class', 'class', 'label', 'Label', 'y', 'Y', 
            'outcome', 'result', 'diagnosis', 'approved', 
            'default', 'risk', 'loan_status', 'hypertension',
            'bp_status', 'addiction', 'addicted', 'phone_addiction'
        ]
        
        for col in possible_targets:
            if col in df.columns:
                df['target'] = df[col]
                break
        
        if 'target' not in df.columns:
            last_col = df.columns[-1]
            if df[last_col].nunique() <= 10:
                df['target'] = df[last_col]
        
        # CORREÇÃO: Verificar se target é contínuo e converter se necessário
        if 'target' in df.columns:
            self._fix_target_column(df)
        
        self._create_target_names(df)

    def _fix_target_column(self, df):
        """
        NOVA FUNÇÃO: Corrige problemas com a coluna target
        """
        if 'target' not in df.columns:
            return
            
        target_col = df['target']
        
        # 1. Remover valores nulos do target
        initial_len = len(df)
        df.dropna(subset=['target'], inplace=True)
        if len(df) < initial_len:
            st.info(f"Removidas {initial_len - len(df)} linhas com target nulo")
        
        # 2. Verificar se target é contínuo demais para classificação
        if pd.api.types.is_numeric_dtype(target_col):
            unique_values = target_col.nunique()
            total_values = len(target_col)
            
            # Se tem muitos valores únicos (>10% do total), pode ser contínuo
            if unique_values > max(10, total_values * 0.1):
                st.warning(f"Target parece ser contínuo ({unique_values} valores únicos)")
                
                # Tentar binning para converter em categórico
                if unique_values > 20:
                    # Usar quartis para criar classes
                    df['target'] = pd.qcut(target_col, q=4, labels=['Baixo', 'Médio-Baixo', 'Médio-Alto', 'Alto'], duplicates='drop')
                    st.info("Target convertido em 4 classes usando quartis")
                elif unique_values > 10:
                    # Usar binning simples
                    df['target'] = pd.cut(target_col, bins=5, labels=['Classe_1', 'Classe_2', 'Classe_3', 'Classe_4', 'Classe_5'])
                    st.info("Target convertido em 5 classes usando binning")
        
        # 3. Garantir que target seja categórico
        if df['target'].dtype == 'float64':
            # Se ainda for float, converter para int se possível
            try:
                if df['target'].isna().sum() == 0:
                    df['target'] = df['target'].astype(int)
            except:
                pass

    def _create_target_names(self, df):
        if 'target_name' in df.columns or 'target' not in df.columns:
            return
        
        unique_targets = df['target'].unique()
        
        # Remover valores NaN se existirem
        unique_targets = unique_targets[pd.notna(unique_targets)]
        
        if len(unique_targets) == 0:
            st.error("Nenhum valor válido encontrado na coluna target")
            return
        
        if df['target'].dtype in ['int64', 'float64'] and len(unique_targets) <= 10:
            if len(unique_targets) == 2:
                self._create_binary_mapping(df, unique_targets)
            elif len(unique_targets) == 3:
                self._create_ternary_mapping(df, unique_targets)
            elif len(unique_targets) == 4:
                sorted_targets = sorted(unique_targets)
                mapping = {val: f'Quartil_{i+1}' for i, val in enumerate(sorted_targets)}
                df['target_name'] = df['target'].map(mapping)
            else:
                df['target_name'] = df['target'].astype(str)
        else:
            df['target_name'] = df['target'].astype(str)

    def _create_binary_mapping(self, df, unique_targets):
        min_val, max_val = sorted(unique_targets)
        
        context_mappings = [
            (['credit_score', 'income', 'loan'], {min_val: 'Denied', max_val: 'Approved'}),
            (['bp', 'blood_pressure', 'hypertension'], {min_val: 'Normal', max_val: 'Hypertensive'}),
            (['phone', 'usage', 'addiction'], {min_val: 'Normal', max_val: 'Addicted'})
        ]
        
        for context_cols, mapping in context_mappings:
            if any(col in df.columns for col in context_cols):
                df['target_name'] = df['target'].map(mapping)
                return
        
        df['target_name'] = df['target'].map({min_val: f'Class_{min_val}', max_val: f'Class_{max_val}'})

    def _create_ternary_mapping(self, df, unique_targets):
        sorted_targets = sorted(unique_targets)
        
        if any(col.lower() in ['glucose', 'hba1c', 'diabetes', 'blood_sugar'] for col in df.columns):
            mapping = {
                sorted_targets[0]: 'Normal',
                sorted_targets[1]: 'Pre-Diabetes', 
                sorted_targets[2]: 'Diabetes'
            }
        else:
            mapping = {val: f'Class_{val}' for val in sorted_targets}
        
        df['target_name'] = df['target'].map(mapping)

    def validate_dataset(self, df):
        if df is None:
            return False, "Dataset é None"
        
        if df.empty:
            return False, "Dataset está vazio"
        
        if len(df.columns) < 2:
            return False, "Dataset deve ter pelo menos 2 colunas"
        
        # Verificar se tem colunas datetime não tratadas
        datetime_cols = df.select_dtypes(include=['datetime64']).columns
        if len(datetime_cols) > 0:
            return False, f"Dataset contém colunas datetime não tratadas: {list(datetime_cols)}"
        
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.5:
            return False, "Dataset tem mais de 50% de valores faltantes"
        
        # Verificar se target existe e é válido
        if 'target' in df.columns:
            target_nulls = df['target'].isnull().sum()
            if target_nulls > 0:
                return False, f"Target tem {target_nulls} valores nulos"
                
            if df['target'].nunique() < 2:
                return False, "Target deve ter pelo menos 2 classes diferentes"
        
        return True, "Dataset válido"

def load_dataset(data_source, uploaded_file=None):
    loader = DataLoader()
    return loader.load_dataset(data_source, uploaded_file)