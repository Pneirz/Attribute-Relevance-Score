import numpy as np
from sklearn.metrics import mean_absolute_error, f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from scipy.stats import wilcoxon
from joblib import Parallel, delayed


class ARS:
    """Implementación mejorada del Attribute Relevance Score (ARS).

    La principal diferencia con la versión anterior es el nuevo cálculo del
    umbral de referencia (`acceptable_minimum`), que ahora se basa en la
    diferencia de medianas y un valor ε (epsilon) obtenido a partir del tamaño
    del efecto de Cohen (dz).
    """

    def __init__(self, X, y, objective='regression', model_type='linear', random_state=42):
        self.X = X  # NumPy array
        self.y = y  # NumPy array
        self.objective = objective
        self.model_type = model_type
        self.random_state = random_state

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
            metric_params = {'average': 'micro'}

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

        # Ejecución paralela
        results = Parallel(n_jobs=-1)(
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