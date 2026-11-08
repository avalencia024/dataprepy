# =============================================================================
# normalization.py
# -----------------------------------------------------------------------------
# Normalización y estandarización, para una variable y para un dataset
# completo:
#   - normalize / normalizeData:     lleva los valores al rango [0, 1].
#   - standardize / standardizeData: media 0 y desviación estándar 1.
# =============================================================================

import warnings

import numpy as np

from .dataset import Dataset
from .utils import _checkNumeric, _toDataFrame, _transformNumericColumns, _variance


def normalize(x):
    """Normaliza un vector numérico al rango [0, 1].

    Parameters
    ----------
    x : list, numpy.ndarray o pandas.Series
        Vector numérico.

    Returns
    -------
    numpy.ndarray
        (x - min(x)) / (max(x) - min(x)). Si todos los valores son iguales el
        rango es 0 y no se puede dividir, así que se devuelve un vector de
        ceros y se avisa con un warning.

    Examples
    --------
    >>> dp.normalize([-0.29, 2.11, -3.84, 10.69, 5.30]).round(3)
    array([0.244, 0.409, 0.   , 1.   , 0.629])
    """

    x = _checkNumeric(x, "x")

    min_value = np.min(x)
    max_value = np.max(x)

    if max_value == min_value:
        warnings.warn("La variable es constante: se devuelve un vector de ceros")
        return np.zeros(len(x))

    return (x - min_value) / (max_value - min_value)


def standardize(x):
    """Estandariza un vector numérico (media 0 y desviación estándar 1).

    Parameters
    ----------
    x : list, numpy.ndarray o pandas.Series
        Vector numérico.

    Returns
    -------
    numpy.ndarray
        (x - media) / desviación estándar. Se usa la desviación estándar
        muestral (la raíz de la varianza que divide entre n - 1, como sd en
        R). Si es 0 (todos los valores iguales) o no se puede calcular (un solo
        valor), se devuelve un vector de ceros y se avisa con un warning.

    Examples
    --------
    >>> dp.standardize([1, 2, 3])
    array([-1.,  0.,  1.])
    """

    x = _checkNumeric(x, "x")

    mean_value = np.mean(x)
    sd_value = np.sqrt(_variance(x))

    # Con un solo valor la desviación estándar no se puede calcular (NaN) y con
    # todos los valores iguales vale 0: en los dos casos no se puede dividir
    if np.isnan(sd_value) or sd_value == 0:
        warnings.warn("La desviación estándar es 0: se devuelve un vector de "
                      "ceros")
        return np.zeros(len(x))

    return (x - mean_value) / sd_value


def normalizeData(data):
    """Normaliza todas las variables numéricas de un dataset.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos. Las variables no numéricas no cambian, y en un Dataset la
        variable clase tampoco.

    Returns
    -------
    pandas.DataFrame o Dataset
        Los datos normalizados, del mismo tipo que 'data'.

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [1.0, 3.0, 5.0], "y": [10, 20, 30]})
    >>> dp.normalizeData(datos)["y"].tolist()
    [0.0, 0.5, 1.0]
    """

    # Con un Dataset transformamos solo las variables que no son la clase
    if isinstance(data, Dataset):
        predictors = data.data[data.getPredictorNames()]
        return data._replacePredictors(normalizeData(predictors))

    data = _toDataFrame(data)

    return _transformNumericColumns(data, normalize)


def standardizeData(data):
    """Estandariza todas las variables numéricas de un dataset.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos. Las variables no numéricas no cambian, y en un Dataset la
        variable clase tampoco.

    Returns
    -------
    pandas.DataFrame o Dataset
        Los datos estandarizados, del mismo tipo que 'data'.

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [1.0, 3.0, 5.0], "y": [10, 20, 30]})
    >>> dp.standardizeData(datos)["x"].tolist()
    [-1.0, 0.0, 1.0]
    """

    if isinstance(data, Dataset):
        predictors = data.data[data.getPredictorNames()]
        return data._replacePredictors(standardizeData(predictors))

    data = _toDataFrame(data)

    return _transformNumericColumns(data, standardize)
