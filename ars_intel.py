from sklearnex import patch_sklearn
patch_sklearn()

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from scipy.stats import wilcoxon
from joblib import Parallel, delayed
from tqdm import tqdm


class ARS:
    """Implementación mejorada del Attribute Relevance Score (ARS).

    La principal diferencia con la versión anterior es el nuevo cálculo del
    umbral de referencia (`acceptable_minimum`), que ahora se basa en la
    diferencia de medianas y un valor ε (epsilon) obtenido a partir del tamaño
    del efecto de Cohen (dz).
    """

    def __init__(self, X, y, objective='regression', model_type='linear', random_state=42,
                 n_jobs: int = -1, parallel_backend: str = 'loky'):
        """Constructor de la clase ARS.

        Parámetros
        ----------
        X, y : np.ndarray
            Datos de entrada y variable objetivo.
        objective : str
            'regression' o 'classification'.
        model_type : str
            Tipo de modelo a utilizar ('linear', 'tree', 'knn').
        random_state : int
            Semilla de aleatoriedad global.
        n_jobs : int, opcional (por defecto = -1)
            Número de hilos/procesos a utilizar en la paralelización de *iteraciones*.
        parallel_backend : str, opcional (por defecto = 'loky')
            Backend de joblib a emplear. 'loky' (por defecto) es más robusto y
            evita el Global Interpreter Lock (GIL) de Python, siendo ideal para
            tareas que consumen CPU. Para tareas muy ligeras donde el overhead
            de crear procesos es un problema, 'threading' puede ser una
            alternativa.
        """

        self.X = X  # NumPy array
        self.y = y  # NumPy array
        self.objective = objective
        self.model_type = model_type
        self.random_state = random_state

        # Nuevos parámetros de rendimiento
        self.n_jobs = n_jobs
        self.parallel_backend = parallel_backend

    # ---------------------------------------------------------------------
    # Métodos internos auxiliares
    # ---------------------------------------------------------------------
    def _relevance_score(self, original_median, benchmark):
        """Calcula la puntuación de relevancia.

        Para clasificación (> mejor),   R = (A - B)/(1 - B)
        Para regresión   (< mejor),     R = (B - A)/B
        """
        if self.objective == 'classification':
            return (original_median - benchmark) / (1 - benchmark)
        elif self.objective == 'regression':
            return (benchmark - original_median) / benchmark
        else:
            raise ValueError('Unspecified objective')

    def _acceptable_minimum(self, A: np.ndarray, B: np.ndarray) -> float:
        """Calcula el *acceptable minimum* usando ε = dz·sd.

        1. Se define el vector de diferencias ∆ según el objetivo.
        2. Se calcula dz = media(∆) / sd(∆).
        3. ε = dz * sd = media(∆).
        4. Para clasificación ⇒ threshold = median(B) + ε.
           Para regresión     ⇒ threshold = median(B) - ε.
        """
        median_A = np.median(A)
        median_B = np.median(B)

        if self.objective == 'classification':
            diff = A - B  # mejora positiva
        elif self.objective == 'regression':
            diff = B - A  # mejora positiva (error menor)
        else:
            raise ValueError('Unspecified objective')

        # sd con 1 grado de libertad (coincide con definición de Cohen)
        sd_diff = np.std(diff, ddof=1)
        dz_min=0.2
        epsilon = 0.0 if sd_diff == 0 else np.mean(diff)*dz_min  # = dz*sd

        if self.objective == 'classification':
            return np.round(median_B + epsilon, 4)
        else:  # regression
            return np.round(median_B - epsilon, 4)

    def _get_model(self, iteration):
        # Selección del modelo según el tipo y el objetivo
        if self.objective == 'regression':
            if self.model_type == 'tree':
                return DecisionTreeRegressor(random_state=self.random_state + iteration, criterion='absolute_error')
            elif self.model_type == 'knn':
                return KNeighborsRegressor(n_neighbors=5)
            elif self.model_type == 'linear':
                return LinearRegression()
            else:
                raise ValueError("Invalid model_type for regression. Choose from 'tree', 'knn', or 'linear'.")
        elif self.objective == 'classification':
            if self.model_type == 'tree':
                return DecisionTreeClassifier(random_state=self.random_state + iteration, criterion='entropy')
            elif self.model_type == 'knn':
                return KNeighborsClassifier(n_neighbors=5)
            else:
                raise ValueError("Invalid model_type for classification. Choose from 'tree' or 'knn'.")
        else:
            raise ValueError("Invalid objective. Choose 'regression' or 'classification'.")

    def _run_iteration(self, iteration, stratify=None):
        # Separación de datos
        X_train, X_test, y_train, y_test = train_test_split(
            self.X,
            self.y,
            stratify=stratify,
            train_size=0.6,
            random_state=self.random_state + iteration,
        )

        # Selección de métrica
        if self.objective == 'regression':
            metric = mean_absolute_error
            metric_params = {}
        else:
            metric = f1_score
            metric_params = {'average': 'binary'}

        # Modelo original
        original_model = self._get_model(iteration)
        original_model.fit(X_train, y_train)
        original_pred = original_model.predict(X_test)
        original_score = metric(y_test, original_pred, **metric_params)

        # Modelo sombra (datos barajados)
        shuffled_indices = np.random.RandomState(self.random_state + iteration).permutation(X_train.shape[0])
        X_train_shuffled = X_train[shuffled_indices]
        shadow_model = self._get_model(iteration)
        shadow_model.fit(X_train_shuffled, y_train)
        shadow_pred = shadow_model.predict(X_test)
        shadow_score = metric(y_test, shadow_pred, **metric_params)

        return original_score, shadow_score

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------
    def calculate(self, max_iterations: int = 1000):
        """Devuelve [ARS, mediana_original, threshold]."""
        np.random.seed(self.random_state)

        # Conversión de tipos
        if self.objective == 'regression':
            y = self.y.astype(float)
            stratify = None
        elif self.objective == 'classification':
            y = self.y.astype(str)
            stratify = y
        else:
            raise ValueError('Specify objective')

        # Caso degenerado
        if np.unique(y).size == 1:
            return [0.0, 0.0, 0.0]

        # Ejecución paralela (configurable)
        results = Parallel(n_jobs=self.n_jobs, backend=self.parallel_backend)(
            delayed(self._run_iteration)(i, stratify) for i in range(max_iterations)
        )

        # Separar y convertir
        A, B = map(np.array, zip(*results))

        # Cálculo de métricas
        acceptable_min = self._acceptable_minimum(A, B)
        median_A = np.round(np.median(A), 4)
        relevance = self._relevance_score(median_A, acceptable_min)

        # Test de Wilcoxon
        alternative = 'greater' if self.objective == 'classification' else 'less'
        _, p_value = wilcoxon(A, B, alternative=alternative)

        if median_A < 0 or relevance < 0 or p_value >= 0.05:
            relevance = 0.0

        return [relevance, median_A, acceptable_min, self.objective, self.model_type]

    @staticmethod
    def calculate_all_features(df: pd.DataFrame, target_column: str, objective='regression', 
                             model_type='linear', random_state=42, max_iterations=1000,
                             n_jobs=-1, parallel_backend='loky', show_progress=True):
        """Calcula el ARS para todas las columnas de un DataFrame (versión optimizada).

        Parámetros
        ----------
        df : pd.DataFrame
            DataFrame con las características y la columna objetivo.
        target_column : str
            Nombre de la columna objetivo.
        objective : str
            'regression' o 'classification'.
        model_type : str
            Tipo de modelo a utilizar ('linear', 'tree', 'knn').
        random_state : int
            Semilla de aleatoriedad global.
        max_iterations : int
            Número máximo de iteraciones para el cálculo del ARS.
        n_jobs : int
            Número de hilos/procesos a utilizar para paralelizar entre características.
        parallel_backend : str
            Backend de joblib a emplear.
        show_progress : bool
            Si mostrar barra de progreso (solo cuando n_jobs=1).

        Retorna
        -------
        pd.DataFrame
            DataFrame con los resultados del ARS para cada característica.
            Columnas: ['feature', 'ARS', 'median_original', 'threshold', 'objective', 'model_type']
        """
        
        if target_column not in df.columns:
            raise ValueError(f"La columna objetivo '{target_column}' no existe en el DataFrame")
        
        # Separar características y variable objetivo
        feature_columns = [col for col in df.columns if col != target_column]
        y = df[target_column].values
        
        # Validar que existan características para evaluar
        if len(feature_columns) == 0:
            raise ValueError("No hay características para evaluar (solo existe la columna objetivo)")
        
        # Validación inicial de NaN en variable objetivo
        if pd.isna(y).any():
            raise ValueError("La variable objetivo contiene valores NaN")
        
        # Filtrar características con NaN
        valid_features = []
        for feature in feature_columns:
            if not pd.isna(df[feature]).any():
                valid_features.append(feature)
            else:
                print(f"Advertencia: La característica '{feature}' contiene valores NaN. Saltando...")
        
        if len(valid_features) == 0:
            print("No hay características válidas para evaluar (todas contienen NaN)")
            return pd.DataFrame(columns=['feature', 'ARS', 'median_original', 'threshold', 'objective', 'model_type'])
        
        # Función interna para calcular ARS de una característica
        def _calculate_single_feature(feature):
            try:
                # Crear matriz X con solo la característica actual
                X = df[[feature]].values
                
                # Crear instancia de ARS con n_jobs=1 para evitar nested parallelization
                ars_instance = ARS(
                    X=X, 
                    y=y, 
                    objective=objective, 
                    model_type=model_type,
                    random_state=random_state,
                    n_jobs=1,  # Evitar paralelización anidada
                    parallel_backend=parallel_backend
                )
                
                result = ars_instance.calculate(max_iterations=max_iterations)
                
                return {
                    'feature': feature,
                    'ARS': result[0],
                    'median_original': result[1],
                    'threshold': result[2],
                    'objective': result[3],
                    'model_type': result[4]
                }
                
            except Exception as e:
                print(f"Error al calcular ARS para la característica '{feature}': {e}")
                return None
        
        # Ejecución paralela o secuencial según configuración
        if n_jobs == 1 and show_progress:
            # Ejecución secuencial con barra de progreso
            results = []
            for feature in tqdm(valid_features, desc="Calculando ARS por característica"):
                result = _calculate_single_feature(feature)
                if result is not None:
                    results.append(result)
        else:
            # Ejecución paralela sin barra de progreso (para evitar conflictos)
            if show_progress and n_jobs != 1:
                print(f"Calculando ARS para {len(valid_features)} características en paralelo...")
            
            results = Parallel(n_jobs=n_jobs, backend=parallel_backend)(
                delayed(_calculate_single_feature)(feature) for feature in valid_features
            )
            
            # Filtrar resultados nulos
            results = [r for r in results if r is not None]
        
        # Convertir resultados a DataFrame y ordenar por ARS descendente
        results_df = pd.DataFrame(results)
        
        if not results_df.empty:
            results_df = results_df.sort_values('ARS', ascending=False).reset_index(drop=True)
        
        return results_df 