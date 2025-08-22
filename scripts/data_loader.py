import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_wine, load_breast_cancer
import streamlit as st
import os

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
        self._ensure_target_column(df)
        return df, f"Dataset Personalizado ({uploaded_file.name})"

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
        
        self._ensure_target_column(df)
        return df, name

    def _ensure_target_column(self, df):
        if 'target' in df.columns:
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
        
        self._create_target_names(df)

    def _create_target_names(self, df):
        if 'target_name' in df.columns or 'target' not in df.columns:
            return
        
        unique_targets = df['target'].unique()
        
        if df['target'].dtype in ['int64', 'float64'] and len(unique_targets) <= 10:
            if len(unique_targets) == 2:
                self._create_binary_mapping(df, unique_targets)
            elif len(unique_targets) == 3:
                self._create_ternary_mapping(df, unique_targets)
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
        
        missing_ratio = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_ratio > 0.5:
            return False, "Dataset tem mais de 50% de valores faltantes"
        
        return True, "Dataset válido"

def load_dataset(data_source, uploaded_file=None):
    loader = DataLoader()
    return loader.load_dataset(data_source, uploaded_file)