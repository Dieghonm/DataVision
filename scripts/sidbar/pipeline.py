import streamlit as st
import pandas as pd
import numpy as np
from scripts.data_loader import load_dataset

class PipelineUI:
    def __init__(self):
        self.current_data = None
        self._init_session_state()

    def _init_session_state(self):
        if 'current_data' not in st.session_state:
            st.session_state.current_data = None
        if 'dataset_name' not in st.session_state:
            st.session_state.dataset_name = None
        if 'last_data_source' not in st.session_state:
            st.session_state.last_data_source = None
        if 'last_uploaded_file' not in st.session_state:
            st.session_state.last_uploaded_file = None

    def _load_dataset(self, data_source, uploaded_file=None):
        try:
            df, name = load_dataset(data_source, uploaded_file)
            return df, name
        except Exception as e:
            st.error(f"Erro ao carregar dataset: {str(e)}")
            return None, None

    def _render_algorithm_params(self, algorithm):
        st.sidebar.markdown("---")
        st.sidebar.subheader("Parâmetros do Algoritmo")
        
        if algorithm == "random_forest":
            n_estimators = st.sidebar.slider(
                "N Estimators", 
                min_value=10, max_value=500, value=100, step=10,
                help="Número de árvores na floresta"
            )
            max_depth = st.sidebar.slider(
                "Max Depth", 
                min_value=1, max_value=50, value=None,
                help="Profundidade máxima das árvores (None = sem limite)"
            )
            min_samples_split = st.sidebar.slider(
                "Min Samples Split",
                min_value=2, max_value=20, value=2,
                help="Mínimo de amostras para dividir um nó"
            )
            min_samples_leaf = st.sidebar.slider(
                "Min Samples Leaf",
                min_value=1, max_value=10, value=1,
                help="Mínimo de amostras em uma folha"
            )
            return {
                "n_estimators": n_estimators, 
                "max_depth": max_depth,
                "min_samples_split": min_samples_split,
                "min_samples_leaf": min_samples_leaf,
                "random_state": 42
            }

        elif algorithm == "logistic_regression":
            c_value = st.sidebar.slider(
                "C (Regularização)", 
                min_value=0.001, max_value=100.0, value=1.0, step=0.001,
                help="Força da regularização (valores menores = mais regularização)"
            )
            max_iter = st.sidebar.slider(
                "Max Iterations", 
                min_value=100, max_value=2000, value=1000, step=100,
                help="Número máximo de iterações"
            )
            solver = st.sidebar.selectbox(
                "Solver", 
                ["lbfgs", "liblinear", "saga"],
                help="Algoritmo de otimização"
            )
            return {"C": c_value, "max_iter": max_iter, "solver": solver, "random_state": 42}

        elif algorithm == "svm":
            kernel = st.sidebar.selectbox(
                "Kernel", 
                ["linear", "rbf", "poly", "sigmoid"],
                help="Tipo de kernel para o SVM"
            )
            c_value = st.sidebar.slider(
                "C (Regularização)", 
                min_value=0.001, max_value=100.0, value=1.0, step=0.001,
                help="Parâmetro de regularização"
            )
            gamma = st.sidebar.selectbox(
                "Gamma",
                ["scale", "auto"],
                help="Coeficiente do kernel"
            )
            return {"kernel": kernel, "C": c_value, "gamma": gamma, "random_state": 42}
            
        elif algorithm == "xgboost":
            n_estimators = st.sidebar.slider("N Estimators", 10, 500, 100, 10)
            max_depth = st.sidebar.slider("Max Depth", 1, 15, 6)
            learning_rate = st.sidebar.slider("Learning Rate", 0.01, 0.3, 0.1, 0.01)
            return {
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                "random_state": 42
            }

        return {}

    def _execute_pipeline(self, config):
        try:
            st.success("Pipeline iniciado!")
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            pipeline_steps = {
                "Carregando dados...": self._load_data,
                "Analisando qualidade dos dados...": self._analyze_data,
                "Preprocessando dados...": self._preprocess_data,
                "Selecionando features...": self._feature_selection,
                "Balanceando classes...": self._balance_classes,
                "Dividindo dataset...": self._split_dataset,
                "Treinando modelo...": self._train_model,
                "Avaliando modelo...": self._evaluate_model,
                "Otimizando hiperparâmetros...": self._hyperparameter_tuning,
                "Salvando resultados...": self._save_results
            }
            
            selected_steps = {}
            for step_name, step_function in pipeline_steps.items():
                step_key = step_name.split("...")[-1].strip()
                
                if "Carregando dados" in step_name or "load_data" in config.get("steps", []):
                    selected_steps[step_name] = step_function
                elif "Analisando" in step_name and ("analyze_data" in config.get("steps", []) or len(config.get("steps", [])) > 3):
                    selected_steps[step_name] = step_function
                elif "Preprocessando" in step_name and ("preprocess_data" in config.get("steps", []) or len(config.get("steps", [])) > 3):
                    selected_steps[step_name] = step_function
                elif "Selecionando features" in step_name and config.get("feature_selection", False):
                    selected_steps[step_name] = step_function
                elif "Balanceando classes" in step_name and config.get("balance_classes", False):
                    selected_steps[step_name] = step_function
                elif "Dividindo dataset" in step_name or "train_model" in config.get("steps", []):
                    selected_steps[step_name] = step_function
                elif "Treinando modelo" in step_name and ("train_model" in config.get("steps", []) or len(config.get("steps", [])) > 1):
                    selected_steps[step_name] = step_function
                elif "Avaliando modelo" in step_name and ("evaluate_model" in config.get("steps", []) or len(config.get("steps", [])) > 2):
                    selected_steps[step_name] = step_function
                elif "Otimizando hiperparâmetros" in step_name and config.get("tune_hyperparameters", False):
                    selected_steps[step_name] = step_function
                elif "Salvando resultados" in step_name and "save_results" in config.get("steps", []):
                    selected_steps[step_name] = step_function
            
            pipeline_data = {}
            steps_container = st.container()
            
            for i, (step_name, step_function) in enumerate(selected_steps.items()):
                try:
                    progress = (i + 1) / len(selected_steps)
                    status_text.text(f"Executando: {step_name}")
                    progress_bar.progress(progress)
                    
                    with steps_container:
                        step_placeholder = st.empty()
                        step_placeholder.info(f"Executando: {step_name}")
                    
                    pipeline_data = step_function(config, pipeline_data)
                    
                    with steps_container:
                        step_placeholder.success(f"{step_name.replace('...', ' concluído!')}")
                    
                except Exception as step_error:
                    with steps_container:
                        step_placeholder.error(f"Erro em {step_name}: {str(step_error)}")
                    st.error(f"Erro na etapa '{step_name}': {str(step_error)}")
                    raise step_error
            
            progress_bar.progress(1.0)
            status_text.text("Pipeline concluído com sucesso!")
            
            with st.container():
                st.success("Pipeline Executado com Sucesso!")
                if 'evaluation_results' in pipeline_data:
                    accuracy = pipeline_data['evaluation_results'].get('accuracy', 0)
                    f1 = pipeline_data['evaluation_results'].get('f1_score', 0)
                    st.info(f"Accuracy Final: {accuracy:.4f} | F1-Score: {f1:.4f}")
            
            self._display_results(config, pipeline_data)
            
        except Exception as e:
            st.error(f"Erro na execução do pipeline: {str(e)}")
            return None
        
        return pipeline_data

    def _load_data(self, config, pipeline_data):
        try:
            if 'DF' not in st.session_state or st.session_state.DF is None:
                raise ValueError("Nenhum DataFrame encontrado no session_state. Faça upload de um arquivo primeiro.")
            
            data = st.session_state.DF.copy()
            
            if data.empty:
                raise ValueError("DataFrame está vazio.")
            
            pipeline_data['raw_data'] = data
            pipeline_data['data_shape'] = data.shape
            pipeline_data['column_names'] = list(data.columns)
            
            st.info(f"Dados carregados: {data.shape[0]} amostras, {data.shape[1]} features")
            st.write(f"Colunas disponíveis: {', '.join(data.columns[:10])}{'...' if len(data.columns) > 10 else ''}")
            
        except Exception as e:
            raise Exception(f"Falha ao carregar dados: {str(e)}")
        
        return pipeline_data

    def _analyze_data(self, config, pipeline_data):
        try:
            data = pipeline_data['raw_data'].copy()
            
            missing_values = data.isnull().sum()
            duplicates = data.duplicated().sum()
            
            target_column = None
            possible_targets = ['target', 'target_name', 'class', 'label', 'y']
            for col in possible_targets:
                if col in data.columns:
                    target_column = col
                    break
            if target_column is None:
                target_column = config.get("target_column", data.columns[-1])
            
            # Correção: Se target_column for 'target_name', usar 'target' se disponível
            if target_column == 'target_name' and 'target' in data.columns:
                target_column = 'target'
            
            class_distribution = data[target_column].value_counts()
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
            
        except Exception as e:
            raise Exception(f"Falha na análise dos dados: {str(e)}")
        
        return pipeline_data

    def _preprocess_data(self, config, pipeline_data):
        try:
            data = pipeline_data['raw_data'].copy()
            analysis = pipeline_data.get('data_analysis', {})
            target_column = analysis.get('target_column')
            
            # Remove colunas desnecessárias (mantém apenas target numérico)
            if 'target_name' in data.columns and 'target' in data.columns:
                data = data.drop(columns=['target_name'])
                st.write("Coluna target_name removida")
            
            if analysis.get('missing_values'):
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
                
                st.write("Valores ausentes tratados")
            
            if analysis.get('duplicates', 0) > 0:
                data = data.drop_duplicates()
                st.write("Duplicatas removidas")
            
            if config.get("remove_outliers", False):
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
            
            categorical_columns = data.select_dtypes(include=['object']).columns
            categorical_columns = [col for col in categorical_columns if col != target_column]
            
            if len(categorical_columns) > 0:
                encoding_method = config.get("encoding_method", "label")
                
                if encoding_method == "label":
                    from sklearn.preprocessing import LabelEncoder
                    for col in categorical_columns:
                        le = LabelEncoder()
                        data[col] = le.fit_transform(data[col])
                        pipeline_data[f'encoder_{col}'] = le
                
                elif encoding_method == "onehot":
                    data = pd.get_dummies(data, columns=categorical_columns, prefix=categorical_columns)
                
                st.write(f"Encoding categórico ({encoding_method}) aplicado")
            
            scaling_method = config.get("scaling", "standard")
            if scaling_method != "none":
                numeric_columns = data.select_dtypes(include=['number']).columns
                numeric_columns = [col for col in numeric_columns if col != target_column]
                
                if len(numeric_columns) > 0:
                    if scaling_method == "standard":
                        from sklearn.preprocessing import StandardScaler
                        scaler = StandardScaler()
                    elif scaling_method == "minmax":
                        from sklearn.preprocessing import MinMaxScaler
                        scaler = MinMaxScaler()
                    elif scaling_method == "robust":
                        from sklearn.preprocessing import RobustScaler
                        scaler = RobustScaler()
                    
                    data[numeric_columns] = scaler.fit_transform(data[numeric_columns])
                    pipeline_data['scaler'] = scaler
                    st.write(f"Normalização ({scaling_method}) aplicada")
            
            pipeline_data['processed_data'] = data
            st.info(f"Preprocessamento concluído. Shape: {data.shape}")
            
        except Exception as e:
            raise Exception(f"Falha no preprocessamento: {str(e)}")
        
        return pipeline_data

    def _feature_selection(self, config, pipeline_data):
        try:
            if not config.get("feature_selection", False):
                pipeline_data['selected_data'] = pipeline_data['processed_data']
                return pipeline_data
            
            data = pipeline_data['processed_data'].copy()
            target_column = pipeline_data['data_analysis']['target_column']
            
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            selection_method = config.get("selection_method", "selectkbest")
            n_features = config.get("n_features", min(10, len(X.columns)))
            
            if selection_method == "selectkbest":
                from sklearn.feature_selection import SelectKBest, f_classif
                selector = SelectKBest(score_func=f_classif, k=n_features)
                X_selected = selector.fit_transform(X, y)
                selected_features = X.columns[selector.get_support()].tolist()
                
            elif selection_method == "rfe":
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
            
            st.write(f"{len(selected_features)} features selecionadas: {', '.join(selected_features[:5])}{'...' if len(selected_features) > 5 else ''}")
            
        except Exception as e:
            pipeline_data['selected_data'] = pipeline_data['processed_data']
            st.warning(f"Seleção de features falhou, usando todas as features: {str(e)}")
        
        return pipeline_data

    def _balance_classes(self, config, pipeline_data):
        try:
            if not config.get("balance_classes", False):
                pipeline_data['balanced_data'] = pipeline_data['selected_data']
                return pipeline_data
            
            data = pipeline_data['selected_data'].copy()
            analysis = pipeline_data['data_analysis']
            
            if not analysis.get('is_imbalanced', False):
                pipeline_data['balanced_data'] = data
                st.write("Dataset já está balanceado")
                return pipeline_data
            
            target_column = analysis['target_column']
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            balance_method = config.get("balance_method", "smote")
            
            if balance_method == "smote":
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=42)
                    X_balanced, y_balanced = smote.fit_resample(X, y)
                    
                    balanced_data = pd.DataFrame(X_balanced, columns=X.columns)
                    balanced_data[target_column] = y_balanced
                    
                    pipeline_data['balanced_data'] = balanced_data
                    st.write(f"SMOTE aplicado. Shape: {balanced_data.shape}")
                    
                except ImportError:
                    st.warning("imblearn não instalado. Usando class_weight='balanced' no modelo.")
                    pipeline_data['balanced_data'] = data
                    pipeline_data['use_class_weight'] = True
            
            else:
                pipeline_data['balanced_data'] = data
                pipeline_data['use_class_weight'] = True
                st.write("Usando class_weight='balanced'")
            
        except Exception as e:
            pipeline_data['balanced_data'] = pipeline_data['selected_data']
            st.warning(f"Balanceamento falhou, continuando sem balanceamento: {str(e)}")
        
        return pipeline_data

    def _split_dataset(self, config, pipeline_data):
        try:
            from sklearn.model_selection import train_test_split
            from sklearn.preprocessing import LabelEncoder
            
            data = pipeline_data['balanced_data'].copy()
            test_size = config["test_size"]
            target_column = pipeline_data['data_analysis']['target_column']
            
            X = data.drop(columns=[target_column])
            y = data[target_column]
            
            if y.dtype == 'object' or not pd.api.types.is_numeric_dtype(y):
                st.write(f"Target categórico detectado. Aplicando encoding...")
                label_encoder = LabelEncoder()
                y = label_encoder.fit_transform(y)
                pipeline_data['target_encoder'] = label_encoder
                pipeline_data['target_classes'] = label_encoder.classes_
                st.write(f"Classes: {', '.join(label_encoder.classes_)}")
            
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42, stratify=y
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=test_size, random_state=42
                )
            
            pipeline_data.update({
                'X_train': X_train,
                'X_test': X_test,
                'y_train': y_train,
                'y_test': y_test,
                'target_column': target_column,
                'feature_columns': list(X.columns)
            })
            
            st.info(f"Dataset dividido - Treino: {X_train.shape[0]}, Teste: {X_test.shape[0]}")
            st.write(f"Features utilizadas: {len(X.columns)}")
            
        except Exception as e:
            raise Exception(f"Falha na divisão do dataset: {str(e)}")
        
        return pipeline_data

    def _train_model(self, config, pipeline_data):
        try:
            algorithm = config["algorithm"]
            algo_params = config["algo_params"].copy()
            
            if pipeline_data.get('use_class_weight', False):
                if algorithm in ['random_forest', 'logistic_regression']:
                    algo_params['class_weight'] = 'balanced'
            
            X_train = pipeline_data['X_train']
            y_train = pipeline_data['y_train']
            
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                model = RandomForestClassifier(**algo_params)
            elif algorithm == "svm":
                from sklearn.svm import SVC
                model = SVC(**algo_params)
            elif algorithm == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                model = LogisticRegression(**algo_params)
            elif algorithm == "xgboost":
                try:
                    import xgboost as xgb
                    model = xgb.XGBClassifier(**algo_params)
                except ImportError:
                    st.error("XGBoost não instalado. Usando Random Forest.")
                    from sklearn.ensemble import RandomForestClassifier
                    model = RandomForestClassifier(n_estimators=100, random_state=42)
            else:
                raise ValueError(f"Algoritmo não suportado: {algorithm}")
            
            model.fit(X_train, y_train)
            
            pipeline_data['trained_model'] = model
            pipeline_data['algorithm_used'] = algorithm
            
            st.info(f"Modelo {algorithm} treinado com sucesso!")
            
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
                feature_names = pipeline_data['feature_columns']
                
                indices = np.argsort(importances)[-5:][::-1]
                top_features = [(feature_names[i], importances[i]) for i in indices]
                
                st.write("Top 5 Features mais importantes:")
                for feature, importance in top_features:
                    st.write(f"- {feature}: {importance:.4f}")
            
        except Exception as e:
            raise Exception(f"Falha no treinamento: {str(e)}")
        
        return pipeline_data

    def _evaluate_model(self, config, pipeline_data):
        try:
            from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                       f1_score, classification_report, confusion_matrix)
            
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
                from sklearn.model_selection import cross_val_score, StratifiedKFold
                cv = StratifiedKFold(n_splits=config.get("cv_folds", 5), shuffle=True, random_state=42)
                
                try:
                    X_full = pd.concat([pipeline_data['X_train'], pipeline_data['X_test']])
                    y_full = np.concatenate([pipeline_data['y_train'], pipeline_data['y_test']])
                    
                    cv_scores = cross_val_score(model, X_full, y_full, cv=cv, scoring='accuracy')
                    results["cv_mean"] = cv_scores.mean()
                    results["cv_std"] = cv_scores.std()
                    
                    st.write(f"Validação Cruzada: {results['cv_mean']:.4f} (±{results['cv_std']:.4f})")
                except Exception as cv_error:
                    st.warning(f"Validação cruzada falhou: {str(cv_error)}")
            
            pipeline_data['evaluation_results'] = results
            pipeline_data['predictions'] = y_pred
            pipeline_data['confusion_matrix'] = confusion_matrix(y_test, y_pred)
            
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
            
            if len(np.unique(y_test)) <= 10:
                st.write("Matriz de Confusão:")
                st.write(pd.DataFrame(
                    pipeline_data['confusion_matrix'],
                    index=[f"Real_{i}" for i in range(len(pipeline_data['confusion_matrix']))],
                    columns=[f"Pred_{i}" for i in range(len(pipeline_data['confusion_matrix'][0]))]
                ))
            
        except Exception as e:
            raise Exception(f"Falha na avaliação: {str(e)}")
        
        return pipeline_data

    def _hyperparameter_tuning(self, config, pipeline_data):
        try:
            if not config.get("tune_hyperparameters", False):
                return pipeline_data
            
            from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
            
            algorithm = config["algorithm"]
            X_train = pipeline_data['X_train']
            y_train = pipeline_data['y_train']
            
            param_grids = {
                "random_forest": {
                    'n_estimators': [50, 100, 200, 300],
                    'max_depth': [5, 10, 15, 20, None],
                    'min_samples_split': [2, 5, 10],
                    'min_samples_leaf': [1, 2, 4],
                    'max_features': ['sqrt', 'log2']
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
                st.write("⚠️ Tuning não disponível para este algoritmo")
                return pipeline_data
            
            # Cria o modelo base
            if algorithm == "random_forest":
                from sklearn.ensemble import RandomForestClassifier
                base_model = RandomForestClassifier(random_state=42)
            elif algorithm == "logistic_regression":
                from sklearn.linear_model import LogisticRegression
                base_model = LogisticRegression(random_state=42)
            elif algorithm == "svm":
                from sklearn.svm import SVC
                base_model = SVC(random_state=42)
            
            # Adiciona class_weight se necessário
            if pipeline_data.get('use_class_weight', False) and algorithm != 'svm':
                param_grids[algorithm]['class_weight'] = ['balanced']
            
            # Busca randomizada
            cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)  # Reduzido para velocidade
            
            random_search = RandomizedSearchCV(
                base_model,
                param_grids[algorithm],
                n_iter=20,  # Número de combinações a testar
                cv=cv,
                scoring='accuracy',
                random_state=42,
                n_jobs=-1
            )
            
            st.write("🔍 Otimizando hiperparâmetros...")
            random_search.fit(X_train, y_train)
            
            # Atualiza o modelo com os melhores parâmetros
            pipeline_data['trained_model'] = random_search.best_estimator_
            pipeline_data['best_params'] = random_search.best_params_
            pipeline_data['best_score'] = random_search.best_score_
            
            st.write(f"✅ Melhores parâmetros encontrados:")
            st.json(random_search.best_params_)
            st.write(f"🎯 **Melhor Score CV:** {random_search.best_score_:.4f}")
            
        except Exception as e:
            st.warning(f"Otimização de hiperparâmetros falhou: {str(e)}")
        
        return pipeline_data

    def _save_results(self, config, pipeline_data):
        """Salva o modelo em data/models e os resultados em data/results"""
        try:
            import pickle
            import json
            import os
            from datetime import datetime
            
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

    def _display_results(self, config, pipeline_data):
        """Exibe os resultados finais com análises detalhadas"""

        st.subheader(" Resumo Executivo")
        
        if 'evaluation_results' in pipeline_data:
            results = pipeline_data['evaluation_results']
            col1, col2, col3 = st.columns(3)
            
            with col1:
                accuracy = results.get('accuracy', 0)
                color = "normal" if accuracy >= 0.8 else "inverse"
                st.metric(" Accuracy Final", f"{accuracy:.4f}", delta=None)
                
            with col2:
                f1 = results.get('f1_score', 0)
                st.metric(" F1-Score", f"{f1:.4f}")
                
            with col3:
                cv_mean = results.get('cv_mean', 0)
                if cv_mean > 0:
                    st.metric("CV Score", f"{cv_mean:.4f}")
        
        st.subheader(" Recomendações para Melhorar a Performance")
        
        if 'evaluation_results' in pipeline_data:
            accuracy = pipeline_data['evaluation_results'].get('accuracy', 0)
            analysis = pipeline_data.get('data_analysis', {})
            
            recommendations = []
            
            if accuracy < 0.7:
                recommendations.append("🔴 **Accuracy muito baixa (<70%)**: Considere coletar mais dados ou revisar a qualidade dos dados")
            elif accuracy < 0.8:
                recommendations.append("🟡 **Accuracy moderada (70-80%)**: Tente otimização de hiperparâmetros ou feature engineering")
            else:
                recommendations.append("🟢 **Boa accuracy (>80%)**: Modelo performando bem!")
            
            if analysis.get('is_imbalanced', False):
                recommendations.append(" **Dataset desbalanceado**: Ative o balanceamento de classes (SMOTE)")
            
            if len(pipeline_data.get('selected_features', pipeline_data.get('feature_columns', []))) > 20:
                recommendations.append(" **Muitas features**: Considere seleção de features mais agressiva")
            
            if not config.get('tune_hyperparameters', False):
                recommendations.append(" **Hiperparâmetros**: Ative a otimização automática de hiperparâmetros")
            
            for rec in recommendations:
                st.write(rec)
        
        with st.expander(" Configuração Utilizada"):
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
        
        with st.expander(" Resultados Detalhados"):
            if 'evaluation_results' in pipeline_data:
                st.json(pipeline_data['evaluation_results'])
            
            if 'best_params' in pipeline_data:
                st.write("**Melhores Hiperparâmetros:**")
                st.json(pipeline_data['best_params'])
            
            if 'selected_features' in pipeline_data:
                st.write(f" **Features Selecionadas ({len(pipeline_data['selected_features'])}):**")
                st.write(", ".join(pipeline_data['selected_features']))
        
        with st.expander("Arquivos Gerados"):
            if 'saved_files' in pipeline_data:
                for file_type, path in pipeline_data['saved_files'].items():
                    st.write(f"- **{file_type.title()}:** `{path}`")

    def render_pipeline(self):
        st.sidebar.title("Pipeline ML")

        # Configurações Básicas
        st.sidebar.subheader("Configurações Básicas")
        
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

        # Verifica mudança de dados
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

        self.current_data = st.session_state.get('current_data')

        algorithm = st.sidebar.selectbox(
            "Algoritmo:",
            ["random_forest", "logistic_regression", "svm", "xgboost"],
            help="Escolha o algoritmo de ML"
        )

        test_size = st.sidebar.slider(
            "Tamanho do teste:", 
            min_value=0.1, max_value=0.5, value=0.2, step=0.05,
            help="Proporção dos dados para teste"
        )

        with st.sidebar.expander("Melhorias de Performance"):
            tune_hyperparameters = st.checkbox(
                "Otimizar Hiperparâmetros",
                help="Busca automática pelos melhores parâmetros"
            )
            
            feature_selection = st.checkbox(
                "Seleção de Features", 
                help="Seleciona automaticamente as melhores features"
            )
            
            balance_classes = st.checkbox(
                "Balancear Classes",
                help="Aplica SMOTE para datasets desbalanceados"
            )
            
            remove_outliers = st.checkbox(
                "Remover Outliers",
                help="Remove outliers usando método IQR"
            )

        with st.sidebar.expander("🔧 Preprocessamento"):
            scaling = st.selectbox(
                "Normalização:", 
                ["standard", "minmax", "robust", "none"],
                help="Tipo de normalização dos dados"
            )
            
            encoding_method = st.selectbox(
                "Encoding Categórico:",
                ["label", "onehot"],
                help="Método para encoding de variáveis categóricas"
            )
            
            missing_strategy = st.selectbox(
                "Valores Ausentes:",
                ["mean", "median", "mode"],
                help="Estratégia para tratar valores ausentes"
            )

        with st.sidebar.expander("⚙️ Parâmetros do Algoritmo"):
            algo_params = self._render_algorithm_params_simple(algorithm)

        with st.sidebar.expander("🔬 Configurações Avançadas"):
            use_cv = st.checkbox(
                "Validação Cruzada",
                value=True,
                help="Usar cross-validation para avaliação mais robusta"
            )
            
            cv_folds = st.slider(
                "CV Folds:", 
                min_value=3, max_value=10, value=5,
                help="Número de folds para validação cruzada"
            )

            metrics = st.multiselect(
                "Métricas:",
                ["accuracy", "precision", "recall", "f1"],
                default=["accuracy", "f1"],
                help="Métricas de avaliação do modelo"
            )

            if feature_selection:
                selection_method = st.selectbox(
                    "Método de Seleção:",
                    ["selectkbest", "rfe"],
                    help="Algoritmo de seleção de features"
                )
                n_features = st.slider(
                    "Número de Features:",
                    min_value=3, max_value=20, value=10,
                    help="Quantas features selecionar"
                )
            else:
                selection_method = "selectkbest"
                n_features = 10
            
            if balance_classes:
                balance_method = st.selectbox(
                    "Método de Balanceamento:",
                    ["smote", "class_weight"],
                    help="Técnica de balanceamento"
                )
            else:
                balance_method = "smote"

        steps = ["load_data", "analyze_data", "preprocess_data", "train_model", "evaluate_model"]
        if feature_selection:
            steps.insert(-2, "feature_selection")
        if balance_classes:
            steps.insert(-2, "balance_classes")
        if tune_hyperparameters:
            steps.insert(-1, "hyperparameter_tuning")

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
                    "use_cv": use_cv,
                    "metrics": metrics,
                    "steps": steps,
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
                self._execute_pipeline(config)
                st.session_state.categoria = "Model_Run"
        else:
            st.sidebar.warning("⚠️ Selecione um dataset para continuar")

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
            "random_state": 42,
            "feature_selection": feature_selection,
            "balance_classes": balance_classes,
            "tune_hyperparameters": tune_hyperparameters
        }

    def _render_algorithm_params_simple(self, algorithm):
        """Versão simplificada dos parâmetros do algoritmo"""
        if algorithm == "random_forest":
            n_estimators = st.slider("N Estimators", 50, 300, 100, 50)
            max_depth = st.slider("Max Depth", 5, 20, 10)
            return {
                "n_estimators": n_estimators, 
                "max_depth": max_depth,
                "random_state": 42
            }

        elif algorithm == "logistic_regression":
            c_value = st.slider("C (Regularização)", 0.01, 10.0, 1.0, 0.1)
            return {"C": c_value, "max_iter": 1000, "random_state": 42}

        elif algorithm == "svm":
            c_value = st.slider("C", 0.1, 10.0, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["rbf", "linear"])
            return {"kernel": kernel, "C": c_value, "random_state": 42}
            
        elif algorithm == "xgboost":
            n_estimators = st.slider("N Estimators", 50, 300, 100, 50)
            learning_rate = st.slider("Learning Rate", 0.05, 0.3, 0.1, 0.05)
            return {
                "n_estimators": n_estimators,
                "learning_rate": learning_rate,
                "random_state": 42
            }

        return {}

def pipeline_sidebar():
    pipeline_ui = PipelineUI()
    return pipeline_ui.render_pipeline()