import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
from scripts.data_loader import load_dataset

class MLPipeline:
    def __init__(self):
        self._init_session_state()
        self.algorithm_params = {
            "random_forest": {
                "n_estimators": (50, 300, 100, 50),
                "max_depth": (5, 20, 10, 1),
                "min_samples_split": (2, 10, 2, 1),
                "min_samples_leaf": (1, 5, 1, 1)
            },
            "logistic_regression": {
                "C": (0.01, 10.0, 1.0, 0.1),
                "max_iter": (500, 2000, 1000, 100)
            },
            "svm": {
                "C": (0.1, 10.0, 1.0, 0.1),
                "kernel": ["rbf", "linear", "poly"]
            },
            "xgboost": {
                "n_estimators": (50, 300, 100, 50),
                "max_depth": (3, 15, 6, 1),
                "learning_rate": (0.05, 0.3, 0.1, 0.05)
            }
        }

    def _init_session_state(self):
        defaults = {
            'current_data': None,
            'dataset_name': None,
            'last_data_source': None,
            'last_uploaded_file': None,
            'executions': []
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value

    def _load_dataset(self, data_source, uploaded_file=None):
        try:
            return load_dataset(data_source, uploaded_file)
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {str(e)}")
            return None, None

    def _render_algorithm_params(self, algorithm):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Parâmetros do Algoritmo")
        
        if algorithm not in self.algorithm_params:
            return {"random_state": 42}
        
        params = {"random_state": 42}
        config = self.algorithm_params[algorithm]
        
        for param_name, param_config in config.items():
            if isinstance(param_config, tuple) and len(param_config) == 4:
                min_val, max_val, default, step = param_config
                params[param_name] = st.sidebar.slider(
                    param_name.replace('_', ' ').title(),
                    min_value=min_val,
                    max_value=max_val,
                    value=default,
                    step=step
                )
            elif isinstance(param_config, list):
                params[param_name] = st.sidebar.selectbox(
                    param_name.replace('_', ' ').title(),
                    param_config
                )
        
        if algorithm == "logistic_regression":
            params["solver"] = st.sidebar.selectbox("Solver", ["lbfgs", "liblinear", "saga"])
        elif algorithm == "svm":
            params["gamma"] = st.sidebar.selectbox("Gamma", ["scale", "auto"])
        
        return params

    def execute_pipeline(self, config):
        try:
            st.success("Pipeline iniciado!")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Pipeline básico sempre executado
            pipeline_steps = [
                ("Carregando dados...", self._load_data),
                ("Analisando qualidade...", self._analyze_data),
                ("Preprocessando...", self._preprocess_data),
                ("Dividindo dataset...", self._split_dataset),
                ("Treinando modelo...", self._train_model),
                ("Avaliando modelo...", self._evaluate_model),
                ("Salvando resultados...", self._save_results)
            ]
            
            # Adicionar etapas opcionais apenas se ativadas
            if config.get("feature_selection", False):
                pipeline_steps.insert(3, ("Selecionando features...", self._feature_selection))
            
            if config.get("balance_classes", False):
                pipeline_steps.insert(-4, ("Balanceando classes...", self._balance_classes))
            
            if config.get("tune_hyperparameters", False):
                pipeline_steps.insert(-1, ("Otimizando hiperparâmetros...", self._hyperparameter_tuning))
            
            pipeline_data = {}
            
            for i, (step_name, step_function) in enumerate(pipeline_steps):
                try:
                    progress = (i + 1) / len(pipeline_steps)
                    status_text.text(f"Executando: {step_name}")
                    progress_bar.progress(progress)
                    
                    step_placeholder = st.empty()
                    step_placeholder.info(f"Executando: {step_name}")
                    
                    pipeline_data = step_function(config, pipeline_data)
                    
                    step_placeholder.success(f"{step_name.replace('...', ' concluído!')}")
                    
                except Exception as step_error:
                    step_placeholder.error(f"Erro em {step_name}: {str(step_error)}")
                    raise step_error
            
            progress_bar.progress(1.0)
            status_text.text("Pipeline concluído com sucesso!")
            
            self._display_results(config, pipeline_data)
            self._save_execution(config, pipeline_data)
            
        except Exception as e:
            st.error(f"Erro na execução do pipeline: {str(e)}")
            return None
        
        return pipeline_data

    def _load_data(self, config, pipeline_data):
        if 'DF' not in st.session_state or st.session_state.DF is None:
            raise ValueError("Nenhum DataFrame encontrado. Faça upload de um arquivo primeiro.")
        
        data = st.session_state.DF.copy()
        
        if data.empty:
            raise ValueError("DataFrame está vazio.")
        
        pipeline_data.update({
            'raw_data': data,
            'data_shape': data.shape,
            'column_names': list(data.columns)
        })
        
        st.info(f"Dados carregados: {data.shape[0]} amostras, {data.shape[1]} features")
        return pipeline_data

    def _analyze_data(self, config, pipeline_data):
        data = pipeline_data['raw_data'].copy()
        
        missing_values = data.isnull().sum()
        duplicates = data.duplicated().sum()
        
        target_column = self._find_target_column(data, config)
        
        if target_column == 'target_name' and 'target' in data.columns:
            target_column = 'target'
        
        # CORREÇÃO: Verificar se target_column existe
        if target_column not in data.columns:
            raise ValueError(f"Coluna target '{target_column}' não encontrada")
        
        # Filtrar valores nulos do target antes de calcular distribuição
        target_data = data[target_column].dropna()
        
        if len(target_data) == 0:
            raise ValueError("Todos os valores do target são nulos")
        
        class_distribution = target_data.value_counts()
        
        if len(class_distribution) < 2:
            raise ValueError(f"Target deve ter pelo menos 2 classes. Encontradas: {len(class_distribution)}")
        
        class_balance_ratio = class_distribution.min() / class_distribution.max()
        
        pipeline_data['data_analysis'] = {
            'missing_values': missing_values[missing_values > 0].to_dict(),
            'duplicates': duplicates,
            'target_column': target_column,
            'class_distribution': class_distribution.to_dict(),
            'class_balance_ratio': class_balance_ratio,
            'is_imbalanced': class_balance_ratio < 0.5
        }
        
        st.info("Análise dos dados concluída!")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Valores Ausentes", len(missing_values[missing_values > 0]))
        with col2:
            st.metric("Duplicatas", duplicates)
        with col3:
            st.metric("Balanceamento", f"{class_balance_ratio:.2f}")
        
        if class_balance_ratio < 0.5:
            st.warning(f"Dataset desbalanceado detectado! Ratio: {class_balance_ratio:.2f}")
        
        return pipeline_data

    def _find_target_column(self, data, config):
        possible_targets = ['target', 'target_name', 'class', 'label', 'y']
        for col in possible_targets:
            if col in data.columns:
                return col
        return config.get("target_column", data.columns[-1])

    def _preprocess_data(self, config, pipeline_data):
        data = pipeline_data['raw_data'].copy()
        analysis = pipeline_data.get('data_analysis', {})
        target_column = analysis.get('target_column')
        
        # CORREÇÃO ADICIONAL: Verificar e tratar colunas problemáticas
        data = self._handle_problematic_columns(data, target_column)
        
        if 'target_name' in data.columns and 'target' in data.columns:
            data = data.drop(columns=['target_name'])
        
        data = self._handle_missing_values(data, config, target_column)
        data = self._remove_duplicates(data, analysis)
        data = self._handle_outliers(data, config, target_column)
        data = self._encode_categorical(data, config, target_column)
        data = self._scale_features(data, config, target_column, pipeline_data)
        
        pipeline_data['processed_data'] = data
        pipeline_data['selected_data'] = data
        pipeline_data['balanced_data'] = data
        st.info(f"Preprocessamento concluído. Shape: {data.shape}")
        
        return pipeline_data
    def _handle_problematic_columns(self, data, target_column):
        """
        NOVA FUNÇÃO: Trata colunas com tipos problemáticos
        """
        data_cleaned = data.copy()
        columns_to_drop = []
        
        for col in data_cleaned.columns:
            if col == target_column:
                continue
                
            # 1. Verificar colunas datetime que não foram tratadas
            if data_cleaned[col].dtype == 'datetime64[ns]':
                try:
                    # Converter datetime em timestamp numérico
                    data_cleaned[col] = data_cleaned[col].astype('int64') // 10**9  # Converter para seconds
                    st.info(f"Coluna datetime '{col}' convertida para timestamp")
                except:
                    columns_to_drop.append(col)
                    st.warning(f"Coluna datetime '{col}' será removida (não pôde ser convertida)")
            
            # 2. Verificar colunas object com alta cardinalidade
            elif data_cleaned[col].dtype == 'object':
                unique_ratio = data_cleaned[col].nunique() / len(data_cleaned)
                if unique_ratio > 0.8 and data_cleaned[col].nunique() > 50:
                    columns_to_drop.append(col)
                    st.warning(f"Coluna '{col}' removida (alta cardinalidade: {data_cleaned[col].nunique()} valores únicos)")
            
            # 3. Verificar colunas com tipos mistos
            elif data_cleaned[col].dtype == 'object':
                # Tentar converter para numérico
                try:
                    # Primeiro, tentar conversão direta
                    data_cleaned[col] = pd.to_numeric(data_cleaned[col], errors='coerce')
                    if data_cleaned[col].isnull().all():
                        columns_to_drop.append(col)
                        st.warning(f"Coluna '{col}' removida (não pôde ser convertida para numérico)")
                except:
                    columns_to_drop.append(col)
                    st.warning(f"Coluna '{col}' removida (tipo problemático)")
        
        # Remover colunas problemáticas
        if columns_to_drop:
            data_cleaned = data_cleaned.drop(columns=columns_to_drop)
            st.info(f"Removidas {len(columns_to_drop)} colunas problemáticas")
        
        return data_cleaned

    def _handle_missing_values(self, data, config, target_column):
        numeric_columns = data.select_dtypes(include=['number']).columns
        categorical_columns = data.select_dtypes(include=['object']).columns
        
        fill_strategy = config.get("missing_strategy", "mean")
        
        if fill_strategy == "mean":
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
        elif fill_strategy == "median":
            data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].median())
        
        for col in categorical_columns:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].mode()[0])
        
        return data

    def _remove_duplicates(self, data, analysis):
        if analysis.get('duplicates', 0) > 0:
            data = data.drop_duplicates()
            st.write("Duplicatas removidas")
        return data

    def _handle_outliers(self, data, config, target_column):
        if not config.get("remove_outliers", False):
            return data
        
        numeric_columns = data.select_dtypes(include=['number']).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]
        
        outliers_removed = 0
        for col in numeric_columns:
            Q1 = data[col].quantile(0.25)
            Q3 = data[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_before = len(data)
            data = data[(data[col] >= lower_bound) & (data[col] <= upper_bound)]
            outliers_removed += outliers_before - len(data)
        
        if outliers_removed > 0:
            st.write(f"{outliers_removed} outliers removidos")
        
        return data

    def _encode_categorical(self, data, config, target_column):
        categorical_columns = data.select_dtypes(include=['object']).columns
        categorical_columns = [col for col in categorical_columns if col != target_column]
        
        if len(categorical_columns) == 0:
            return data
        
        encoding_method = config.get("encoding_method", "label")
        
        if encoding_method == "label":
            from sklearn.preprocessing import LabelEncoder
            for col in categorical_columns:
                le = LabelEncoder()
                data[col] = le.fit_transform(data[col])
        elif encoding_method == "onehot":
            data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns)
        
        st.write(f"Encoding categórico ({encoding_method}) aplicado")
        return data

    def _scale_features(self, data, config, target_column, pipeline_data):
        scaling_method = config.get("scaling", "standard")
        if scaling_method == "none":
            return data
        
        numeric_columns = data.select_dtypes(include=['number']).columns
        numeric_columns = [col for col in numeric_columns if col != target_column]
        
        if len(numeric_columns) == 0:
            return data
        
        from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
        
        scaler_map = {
            "standard": StandardScaler(),
            "minmax": MinMaxScaler(),
            "robust": RobustScaler()
        }
        
        scaler = scaler_map[scaling_method]
        data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
        pipeline_data['scaler'] = scaler
        
        st.write(f"Normalização ({scaling_method}) aplicada")
        return data

    def _feature_selection(self, config, pipeline_data):
        if not config.get("feature_selection", False):
            # Se não há seleção de features, usar dados processados
            pipeline_data['selected_data'] = pipeline_data.get('processed_data', pipeline_data.get('raw_data'))
            return pipeline_data
        
        data = pipeline_data.get('processed_data', pipeline_data.get('raw_data')).copy()
        target_column = pipeline_data['data_analysis']['target_column']
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        selection_method = config.get("selection_method", "selectkbest")
        n_features = config.get("n_features", min(10, len(X.columns)))
        
        try:
            if selection_method == "selectkbest":
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(score_func=f_classif, k=n_features)
            else:
                from sklearn.feature_selection import RFE
                from sklearn.ensemble import RandomForestClassifier
                estimator = RandomForestClassifier(n_estimators=10, random_state=42)
                selector = RFE(estimator, n_features_to_select=n_features)
            
            X_selected = selector.fit_transform(X, y)
            selected_features = X.columns[selector.get_support()].tolist()
            
            selected_data = pd.DataFrame(X_selected, columns=selected_features, index=X.index)
            selected_data[target_column] = y
            
            pipeline_data['selected_data'] = selected_data
            pipeline_data['selected_features'] = selected_features
            pipeline_data['feature_selector'] = selector
            
            st.write(f"{len(selected_features)} features selecionadas")
            
        except Exception as e:
            pipeline_data['selected_data'] = pipeline_data.get('processed_data', pipeline_data.get('raw_data'))
            st.warning(f"Seleção de features falhou: {str(e)}")
        
        return pipeline_data

    def _balance_classes(self, config, pipeline_data):
        if not config.get("balance_classes", False):
            # Se não há balanceamento, usar dados selecionados
            pipeline_data['balanced_data'] = pipeline_data.get('selected_data', 
                                                              pipeline_data.get('processed_data', 
                                                                               pipeline_data.get('raw_data')))
            return pipeline_data
        
        data = pipeline_data.get('selected_data', 
                                pipeline_data.get('processed_data', 
                                                 pipeline_data.get('raw_data'))).copy()
        analysis = pipeline_data['data_analysis']
        
        if not analysis.get('is_imbalanced', False):
            pipeline_data['balanced_data'] = data
            st.write("Dataset já está balanceado")
            return pipeline_data
        
        target_column = analysis['target_column']
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        try:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(random_state=42)
            X_balanced, y_balanced = smote.fit_resample(X, y)
            
            balanced_data = pd.DataFrame(X_balanced, columns=X.columns)
            balanced_data[target_column] = y_balanced
            
            pipeline_data['balanced_data'] = balanced_data
            st.write(f"SMOTE aplicado. Shape: {balanced_data.shape}")
            
        except ImportError:
            st.warning("imblearn não instalado. Usando class_weight='balanced'")
            pipeline_data['balanced_data'] = data
            pipeline_data['use_class_weight'] = True
        
        return pipeline_data
    
    def _prepare_target_for_training(self, y, pipeline_data):
        """
        NOVA FUNÇÃO: Prepara a variável target para o treinamento
        """
        from sklearn.preprocessing import LabelEncoder
        
        # 1. Remover valores nulos
        if y.isnull().any():
            st.warning(f"Removendo {y.isnull().sum()} valores nulos do target")
            # Isso precisa ser tratado no nível do dataset, não apenas do y
            
        # 2. Verificar se é contínuo demais
        if pd.api.types.is_numeric_dtype(y):
            unique_values = y.nunique()
            
            # Se tem mais de 20 valores únicos, pode ser regressão
            if unique_values > 20:
                st.warning(f"Target tem {unique_values} valores únicos - convertendo para classificação")
                
                # Usar quartis para criar classes
                try:
                    y_binned = pd.qcut(y, q=4, labels=False, duplicates='drop')
                    st.info("Target convertido em 4 classes usando quartis")
                    y = y_binned
                except Exception as e:
                    # Se quartis falharem, usar bins fixos
                    try:
                        y_binned = pd.cut(y, bins=5, labels=False)
                        st.info("Target convertido em 5 classes usando bins fixos")
                        y = y_binned
                    except Exception as e2:
                        st.error(f"Não foi possível converter target: {str(e2)}")
                        raise ValueError("Target não pôde ser preparado para classificação")
        
        # 3. Converter para inteiros se for numérico
        if pd.api.types.is_numeric_dtype(y) and y.dtype != 'int64':
            try:
                y = y.astype('int64')
            except:
                # Se não conseguir converter para int, usar LabelEncoder
                label_encoder = LabelEncoder()
                y = pd.Series(label_encoder.fit_transform(y), index=y.index)
                pipeline_data['target_encoder'] = label_encoder
                pipeline_data['target_classes'] = label_encoder.classes_
        
        # 4. Se for object, usar LabelEncoder
        elif y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
            label_encoder = LabelEncoder()
            y = pd.Series(label_encoder.fit_transform(y), index=y.index)
            pipeline_data['target_encoder'] = label_encoder
            pipeline_data['target_classes'] = label_encoder.classes_
        
        # 5. Verificação final
        if y.nunique() < 2:
            raise ValueError(f"Target deve ter pelo menos 2 classes. Atualmente tem: {y.nunique()}")
        
        return y



    def _split_dataset(self, config, pipeline_data):
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import LabelEncoder
        
        # Garantir que temos os dados para dividir
        data = pipeline_data.get('balanced_data', 
                                pipeline_data.get('selected_data',
                                                pipeline_data.get('processed_data',
                                                                pipeline_data.get('raw_data')))).copy()
        
        if data is None:
            raise ValueError("Nenhum dado disponível para divisão")
        
        test_size = config["test_size"]
        target_column = pipeline_data['data_analysis']['target_column']
        
        # CORREÇÃO: Verificar se target_column ainda existe
        if target_column not in data.columns:
            # Procurar por coluna target alternativa
            if 'target' in data.columns:
                target_column = 'target'
            else:
                raise ValueError(f"Coluna target '{target_column}' não encontrada nos dados")
        
        X = data.drop(columns=[target_column])
        y = data[target_column]
        
        # CORREÇÃO: Tratar target antes da divisão
        y = self._prepare_target_for_training(y, pipeline_data)
        
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=42, stratify=y
            )
        except ValueError as e:
            if "stratify" in str(e):
                # Se stratify falhar, fazer sem estratificação
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            else:
                raise e
        
        pipeline_data.update({
            'X_train': X_train,
            'X_test': X_test,
            'y_train': y_train,
            'y_test': y_test,
            'target_column': target_column,
            'feature_columns': list(X.columns)
        })
        
        st.info(f"Dataset dividido - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")
        return pipeline_data

    def _train_model(self, config, pipeline_data):
        algorithm = config["algorithm"]
        algo_params = config["algo_params"].copy()
        
        if pipeline_data.get('use_class_weight', False) and algorithm in ['random_forest', 'logistic_regression']:
            algo_params['class_weight'] = 'balanced'
        
        X_train = pipeline_data['X_train']
        y_train = pipeline_data['y_train']
        
        model_map = {
            "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
            "svm": ("sklearn.svm", "SVC"),
            "logistic_regression": ("sklearn.linear_model", "LogisticRegression"),
            "xgboost": ("xgboost", "XGBClassifier")
        }
        
        try:
            if algorithm == "xgboost":
                import xgboost as xgb
                model = xgb.XGBClassifier(**algo_params)
            else:
                module_name, class_name = model_map[algorithm]
                module = __import__(module_name, fromlist=[class_name])
                model_class = getattr(module, class_name)
                model = model_class(**algo_params)
            
            model.fit(X_train, y_train)
            
            pipeline_data['trained_model'] = model
            pipeline_data['algorithm_used'] = algorithm
            
            st.info(f"Modelo {algorithm} treinado com sucesso!")
            
            if hasattr(model, 'feature_importances_'):
                self._display_feature_importance(model, pipeline_data['feature_columns'])
            
        except Exception as e:
            raise Exception(f"Falha no treinamento: {str(e)}")
        
        return pipeline_data

    def _display_feature_importance(self, model, feature_names):
        importances = model.feature_importances_
        indices = np.argsort(importances)[-5:][::-1]
        top_features = [(feature_names[i], importances[i]) for i in indices]
        
        st.write("Top 5 Features mais importantes:")
        for feature, importance in top_features:
            st.write(f"- {feature}: {importance:.4f}")

    def _evaluate_model(self, config, pipeline_data):
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                   f1_score, confusion_matrix)
        
        model = pipeline_data['trained_model']
        X_test = pipeline_data['X_test']
        y_test = pipeline_data['y_test']
        
        y_pred = model.predict(X_test)
        
        results = {
            "accuracy": accuracy_score(y_test, y_pred),
            "precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "f1_score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
        
        if config.get("use_cv", True):
            results.update(self._cross_validation(config, pipeline_data))
        
        pipeline_data['evaluation_results'] = results
        pipeline_data['predictions'] = y_pred
        pipeline_data['confusion_matrix'] = confusion_matrix(y_test, y_pred)
        
        self._display_evaluation_results(results, pipeline_data)
        return pipeline_data

    def _cross_validation(self, config, pipeline_data):
        try:
            from sklearn.model_selection import cross_val_score, StratifiedKFold
            
            model = pipeline_data['trained_model']
            cv = StratifiedKFold(n_splits=config.get("cv_folds", 5), shuffle=True, random_state=42)
            
            X_full = pd.concat([pipeline_data['X_train'], pipeline_data['X_test']])
            y_full = np.concatenate([pipeline_data['y_train'], pipeline_data['y_test']])
            
            cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
            
            return {
                "cv_mean": cv_scores.mean(),
                "cv_std": cv_scores.std()
            }
        except Exception as cv_error:
            st.warning(f"Validação cruzada falhou: {str(cv_error)}")
            return {}

    def _display_evaluation_results(self, results, pipeline_data):
        st.info("Avaliação concluída!")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Accuracy", f"{results['accuracy']:.4f}")
        with col2:
            st.metric("Precision", f"{results['precision']:.4f}")
        with col3:
            st.metric("Recall", f"{results['recall']:.4f}")
        with col4:
            st.metric("F1-Score", f"{results['f1_score']:.4f}")
        
        if 'cv_mean' in results:
            st.write(f"Validação Cruzada: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
        
        if len(np.unique(pipeline_data['y_test'])) <= 10:
            st.write("Matriz de Confusão:")
            cm = pipeline_data['confusion_matrix']
            st.write(pd.DataFrame(
                cm,
                index=[f"Real_{i}" for i in range(len(cm))],
                columns=[f"Pred_{i}" for i in range(len(cm[0]))]
            ))

    def _hyperparameter_tuning(self, config, pipeline_data):
        if not config.get("tune_hyperparameters", False):
            return pipeline_data
        
        try:
            from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
            
            algorithm = config["algorithm"]
            X_train = pipeline_data['X_train']
            y_train = pipeline_data['y_train']
            
            param_grids = {
                "random_forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4]
                },
                "logistic_regression": {
                    'C': [0.001, 0.01, 0.1, 1, 10, 100],
                    'solver': ['liblinear', 'lbfgs'],
                    'max_iter': [100, 500, 1000]
                },
                "svm": {
                    'C': [0.1, 1, 10, 100],
                    'kernel': ['linear', 'rbf'],
                    'gamma': ['scale', 'auto']
                }
            }
            
            if algorithm not in param_grids:
                st.write("Tuning não disponível para este algoritmo")
                return pipeline_data
            
            model_map = {
                "random_forest": ("sklearn.ensemble", "RandomForestClassifier"),
                "logistic_regression": ("sklearn.linear_model", "LogisticRegression"),
                "svm": ("sklearn.svm", "SVC")
            }
            
            module_name, class_name = model_map[algorithm]
            module = __import__(module_name, fromlist=[class_name])
            model_class = getattr(module, class_name)
            base_model = model_class(random_state=42)
            
            if pipeline_data.get('use_class_weight', False) and algorithm != 'svm':
                param_grids[algorithm]['class_weight'] = ['balanced']
            
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
            
            random_search = RandomizedSearchCV(
                base_model,
                param_grids[algorithm],
                n_iter=20,
                cv=cv,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1
            )
            
            st.write("Otimizando hiperparâmetros...")
            random_search.fit(X_train, y_train)
            
            pipeline_data['trained_model'] = random_search.best_estimator_
            pipeline_data['best_params'] = random_search.best_params_
            pipeline_data['best_score'] = random_search.best_score_
            
            st.write("Melhores parâmetros encontrados:")
            st.json(random_search.best_params_)
            st.write(f"Melhor Score CV: {random_search.best_score_:.4f}")
            
        except Exception as e:
            st.warning(f"Otimização de hiperparâmetros falhou: {str(e)}")
        
        return pipeline_data

    def _save_results(self, config, pipeline_data):
        try:
            import pickle
            import json
            import os
            
            os.makedirs("data/models", exist_ok=True)
            os.makedirs("data/results", exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            model_path = os.path.join("data", "models", f"model_{timestamp}.pkl")
            with open(model_path, 'wb') as f:
                pickle.dump(pipeline_data['trained_model'], f)
            
            results_path = os.path.join("data", "results", f"results_{timestamp}.json")
            results_to_save = {
                "config": config,
                "evaluation_results": pipeline_data['evaluation_results'],
                "data_shape": pipeline_data['data_shape'],
                "algorithm_used": pipeline_data['algorithm_used'],
                "best_params": pipeline_data.get('best_params', {}),
                "data_analysis": pipeline_data.get('data_analysis', {}),
                "selected_features": pipeline_data.get('selected_features', []),
                "timestamp": timestamp
            }
            
            with open(results_path, 'w') as f:
                json.dump(results_to_save, f, indent=2, default=str)
            
            pipeline_data['saved_files'] = {
                'model': model_path,
                'results': results_path
            }
            
            st.info(f"Resultados salvos em:\n- {model_path}\n- {results_path}")
            
        except Exception as e:
            raise Exception(f"Falha ao salvar resultados: {str(e)}")
        
        return pipeline_data

    def _save_execution(self, config, pipeline_data):
        if 'executions' not in st.session_state:
            st.session_state.executions = []
        
        execution = {
            'timestamp': datetime.now(),
            'config': config,
            'results': {
                'status': 'success',
                'evaluation': {
                    'metrics': pipeline_data.get('evaluation_results', {})
                }
            }
        }
        
        st.session_state.executions.append(execution)

    def _display_results(self, config, pipeline_data):
        st.subheader("Resumo Executivo")
        
        if 'evaluation_results' in pipeline_data:
            results = pipeline_data['evaluation_results']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = results.get('accuracy', 0)
                st.metric("Accuracy Final", f"{accuracy:.4f}")
                
            with col2:
                f1 = results.get('f1_score', 0)
                st.metric("F1-Score", f"{f1:.4f}")
                
            with col3:
                cv_mean = results.get('cv_mean', 0)
                if cv_mean > 0:
                    st.metric("CV Score", f"{cv_mean:.4f}")
        
        self._display_recommendations(config, pipeline_data)
        self._display_detailed_results(config, pipeline_data)

    def _display_recommendations(self, config, pipeline_data):
        st.subheader("Recomendações para Melhorar a Performance")
        
        if 'evaluation_results' in pipeline_data:
            accuracy = pipeline_data['evaluation_results'].get('accuracy', 0)
            analysis = pipeline_data.get('data_analysis', {})
            
            recommendations = []
            
            if accuracy < 0.7:
                recommendations.append("Accuracy muito baixa (<70%): Considere coletar mais dados ou revisar a qualidade")
            elif accuracy < 0.8:
                recommendations.append("Accuracy moderada (70-80%): Tente otimização de hiperparâmetros ou feature engineering")
            else:
                recommendations.append("Boa accuracy (>80%): Modelo performando bem!")
            
            if analysis.get('is_imbalanced', False):
                recommendations.append("Dataset desbalanceado: Ative o balanceamento de classes (SMOTE)")
            
            if len(pipeline_data.get('selected_features', pipeline_data.get('feature_columns', []))) > 20:
                recommendations.append("Muitas features: Considere seleção de features mais agressiva")
            
            if not config.get('tune_hyperparameters', False):
                recommendations.append("Hiperparâmetros: Ative a otimização automática de hiperparâmetros")
            
            for rec in recommendations:
                st.write(f"- {rec}")

    def _display_detailed_results(self, config, pipeline_data):
        with st.expander("Configuração Utilizada"):
            col1, col2 = st.columns(2)
            with col1:
                st.json({
                    "data_source": config["data_source"],
                    "algorithm": config["algorithm"],
                    "test_size": config["test_size"],
                    "scaling": config["scaling"]
                })
            with col2:
                st.json({
                    "metrics": config["metrics"],
                    "feature_selection": config.get("feature_selection", False),
                    "balance_classes": config.get("balance_classes", False),
                    "tune_hyperparameters": config.get("tune_hyperparameters", False)
                })
        
        with st.expander("Resultados Detalhados"):
            if 'evaluation_results' in pipeline_data:
                st.json(pipeline_data['evaluation_results'])
            
            if 'best_params' in pipeline_data:
                st.write("**Melhores Hiperparâmetros:**")
                st.json(pipeline_data['best_params'])
            
            if 'selected_features' in pipeline_data:
                st.write(f"**Features Selecionadas ({len(pipeline_data['selected_features'])}):**")
                st.write(", ".join(pipeline_data['selected_features']))
        
        with st.expander("Arquivos Gerados"):
            if 'saved_files' in pipeline_data:
                for file_type, path in pipeline_data['saved_files'].items():
                    st.write(f"- **{file_type.title()}:** `{path}`")

    def render_sidebar(self):
        st.sidebar.title("Pipeline ML")
        st.sidebar.subheader("Configurações Básicas")
        
        data_source = st.sidebar.selectbox(
            "Fonte de dados:",
            ["upload", "iris", "wine", "breast_cancer", "Credit", "Hipertension", "Phone addiction"],
            key="data_source_select"
        )

        uploaded_file = None
        if data_source == "upload":
            uploaded_file = st.sidebar.file_uploader(
                "Upload CSV:", 
                type=['csv'],
                key="file_upload"
            )

        data_changed = (
            data_source != st.session_state.get('last_data_source') or
            uploaded_file != st.session_state.get('last_uploaded_file')
        )

        if data_changed:
            with st.spinner("Carregando dataset..."):
                current_data, dataset_name = self._load_dataset(data_source, uploaded_file)
                st.session_state.current_data = current_data
                st.session_state.dataset_name = dataset_name
                st.session_state.last_data_source = data_source
                st.session_state.last_uploaded_file = uploaded_file

        current_data = st.session_state.get('current_data')

        algorithm = st.sidebar.selectbox(
            "Algoritmo:",
            ["random_forest", "logistic_regression", "svm", "xgboost"]
        )

        test_size = st.sidebar.slider(
            "Tamanho do teste:", 
            min_value=0.1, max_value=0.5, value=0.2, step=0.05
        )

        with st.sidebar.expander("Melhorias de Performance"):
            tune_hyperparameters = st.checkbox("Otimizar Hiperparâmetros")
            feature_selection = st.checkbox("Seleção de Features")
            balance_classes = st.checkbox("Balancear Classes")
            remove_outliers = st.checkbox("Remover Outliers")

        with st.sidebar.expander("Preprocessamento"):
            scaling = st.selectbox("Normalização:", ["standard", "minmax", "robust", "none"])
            encoding_method = st.selectbox("Encoding Categórico:", ["label", "onehot"])
            missing_strategy = st.selectbox("Valores Ausentes:", ["mean", "median", "mode"])

        with st.sidebar.expander("Parâmetros do Algoritmo"):
            algo_params = self._render_algorithm_params(algorithm)

        with st.sidebar.expander("Configurações Avançadas"):
            use_cv = st.checkbox("Validação Cruzada", value=True)
            cv_folds = st.slider("CV Folds:", min_value=3, max_value=10, value=5)
            metrics = st.multiselect(
                "Métricas:",
                ["accuracy", "precision", "recall", "f1"],
                default=["accuracy", "f1"]
            )

            if feature_selection:
                selection_method = st.selectbox("Método de Seleção:", ["selectkbest", "rfe"])
                n_features = st.slider("Número de Features:", min_value=3, max_value=20, value=10)
            else:
                selection_method = "selectkbest"
                n_features = 10
            
            if balance_classes:
                balance_method = st.selectbox("Método de Balanceamento:", ["smote", "class_weight"])
            else:
                balance_method = "smote"

        st.sidebar.markdown("---")
        
        if current_data is not None:
            if st.sidebar.button("Executar Pipeline", type="primary", use_container_width=True):
                config = {
                    "data_source": data_source,
                    "uploaded_file": uploaded_file,
                    "algorithm": algorithm,
                    "algo_params": algo_params,
                    "test_size": test_size,
                    "cv_folds": cv_folds,
                    "use_cv": use_cv,
                    "metrics": metrics,
                    "scaling": scaling,
                    "encoding_method": encoding_method,
                    "missing_strategy": missing_strategy,
                    "remove_outliers": remove_outliers,
                    "feature_selection": feature_selection,
                    "selection_method": selection_method,
                    "n_features": n_features,
                    "balance_classes": balance_classes,
                    "balance_method": balance_method,
                    "tune_hyperparameters": tune_hyperparameters,
                    "random_state": 42
                }
                
                self.execute_pipeline(config)
                st.session_state.categoria = "Model_Run"
        else:
            st.sidebar.warning("Selecione um dataset para continuar")

        return {
            "data_source": data_source,
            "uploaded_file": uploaded_file,
            "algorithm": algorithm,
            "algo_params": algo_params,
            "test_size": test_size,
            "cv_folds": cv_folds,
            "metrics": metrics,
            "scaling": scaling,
            "random_state": 42,
            "feature_selection": feature_selection,
            "balance_classes": balance_classes,
            "tune_hyperparameters": tune_hyperparameters
        }

def pipeline_sidebar():
    pipeline = MLPipeline()
    return pipeline.render_sidebar()