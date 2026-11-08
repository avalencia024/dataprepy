# =============================================================================
# correlation.py
# -----------------------------------------------------------------------------
# Relación por pares entre las variables de un dataset:
#   - correlation:         correlación de Pearson entre dos variables numéricas.
#   - mutualInformation:   información mutua entre dos variables categóricas.
#   - computeCorrelations: matriz de correlaciones (variables numéricas) y de
#                          información mutua (variables categóricas).
# =============================================================================

import numpy as np
import pandas as pd

from .dataset import Dataset
from .metrics import entropy
from .utils import (_checkNumeric, _isCategorical, _isCategory, _isNumeric,
                    _pairwiseMatrix, _toDataFrame)


def correlation(x, y):
    """Correlación de Pearson entre dos variables numéricas.

    Parameters
    ----------
    x, y : list, numpy.ndarray o pandas.Series
        Vectores numéricos de la misma longitud.

    Returns
    -------
    float
        Valor entre -1 y 1. Si alguna variable es constante la correlación no
        está definida y se devuelve NaN.

    Notes
    -----
    r = sum((x - media(x)) * (y - media(y))) /
        sqrt(sum((x - media(x))^2) * sum((y - media(y))^2))

    Examples
    --------
    >>> dp.correlation([1, 2, 3], [3, 2, 1])
    -1.0
    """

    x = _checkNumeric(x, "x")
    y = _checkNumeric(y, "y")

    if len(x) != len(y):
        raise ValueError("'x' e 'y' deben tener la misma longitud")

    # Restamos la media a cada variable
    x_centered = x - np.mean(x)
    y_centered = y - np.mean(y)

    denominator = np.sqrt(np.sum(x_centered ** 2) * np.sum(y_centered ** 2))

    # Si una de las variables es constante el denominador es 0
    if denominator == 0:
        return np.nan

    return float(np.sum(x_centered * y_centered) / denominator)


def mutualInformation(x, y):
    """Información mutua (en bits) entre dos variables categóricas.

    Parameters
    ----------
    x, y : pandas.Categorical o pandas.Series de tipo category
        Variables de la misma longitud.

    Returns
    -------
    float
        Valor mayor o igual que 0. Vale 0 si las variables son independientes
        y la información mutua de una variable consigo misma es su entropía.

    Notes
    -----
    Se calcula con la entropía: I(X; Y) = H(X) + H(Y) - H(X, Y), donde
    H(X, Y) es la entropía de la variable que junta los valores de las dos
    (cada pareja de valores es un valor de la variable conjunta).

    Examples
    --------
    >>> x = pd.Categorical(["a", "a", "b", "b"])
    >>> y = pd.Categorical(["u", "v", "u", "v"])
    >>> dp.mutualInformation(x, y)
    0.0
    """

    if not _isCategory(x) or not _isCategory(y):
        raise TypeError("'x' e 'y' deben ser variables categóricas")

    if len(x) != len(y):
        raise ValueError("'x' e 'y' deben tener la misma longitud")

    x = pd.Categorical(x)
    y = pd.Categorical(y)

    # Variable conjunta: para cada observación juntamos en un texto el número
    # de categoría de x y el de y (por ejemplo "0_2"). Usamos los números
    # (codes) y no los nombres para que dos parejas distintas nunca den el
    # mismo texto.
    joint = [f"{code_x}_{code_y}" for code_x, code_y in zip(x.codes, y.codes)]
    joint = pd.Categorical(joint)

    mi_value = entropy(x) + entropy(y) - entropy(joint)

    # Por los redondeos puede salir algo como -1e-16 en lugar de 0
    return max(0.0, mi_value)


def computeCorrelations(data):
    """Correlación e información mutua por pares entre las variables.

    Entre dos variables numéricas se calcula la correlación de Pearson y entre
    dos categóricas la información mutua.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos. En un Dataset también se incluye la variable clase.

    Returns
    -------
    correlation_matrix : pandas.DataFrame o None
        Correlaciones entre las variables numéricas (None si no hay ninguna).
    mutual_information_matrix : pandas.DataFrame o None
        Información mutua entre las variables categóricas (None si no hay
        ninguna).

    Notes
    -----
    Como la correlación y la información mutua miden cosas distintas y tienen
    escalas diferentes, se devuelven en dos matrices. Los pares de una
    variable numérica y otra categórica no se comparan; para hacerlo se puede
    discretizar antes la numérica con discretizeData.

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [2.0, 4.0, 6.0]})
    >>> correlations, mutual_info = dp.computeCorrelations(datos)
    >>> float(correlations.loc["x", "y"])
    1.0
    """

    if isinstance(data, Dataset):
        data = data.data

    data = _toDataFrame(data)

    # Separamos las variables según su tipo
    numeric_columns = [c for c in data.columns if _isNumeric(data[c])]
    categorical_columns = [c for c in data.columns if _isCategorical(data[c])]

    correlation_matrix = None
    mutual_information_matrix = None

    if len(numeric_columns) > 0:
        correlation_matrix = _pairwiseMatrix(data[numeric_columns], correlation)

    if len(categorical_columns) > 0:
        # Las variables booleanas se pasan a category para poder usar entropy
        categorical_data = data[categorical_columns].astype("category")
        mutual_information_matrix = _pairwiseMatrix(categorical_data,
                                                    mutualInformation)

    return correlation_matrix, mutual_information_matrix
