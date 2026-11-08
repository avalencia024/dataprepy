# =============================================================================
# utils.py
# -----------------------------------------------------------------------------
# Funciones auxiliares que usan el resto de ficheros del paquete. Su nombre
# empieza por "_" para indicar que son internas (no son funciones pensadas
# para que las use directamente quien instala el paquete).
# =============================================================================

import warnings

import numpy as np
import pandas as pd


def _checkNumeric(x, arg_name="x"):
    """Comprueba que 'x' sea un vector numérico sin valores NaN ni infinitos y
    lo devuelve convertido en un array de numpy. Si no es válido, lanza un
    error.

    Entradas:
        x         Lista, array de numpy o Series de pandas.
        arg_name  Nombre del argumento, para que el mensaje de error sea claro.
    """

    x = np.array(x)

    # Tiene que ser un vector (una sola dimensión), no un número suelto ni una
    # matriz
    if x.ndim != 1:
        raise ValueError(f"'{arg_name}' debe ser un vector")

    # np.number incluye los enteros y los reales, pero no los booleanos
    if not np.issubdtype(x.dtype, np.number):
        raise TypeError(f"'{arg_name}' debe ser un vector numérico")

    if len(x) == 0:
        raise ValueError(f"'{arg_name}' no puede estar vacío")

    # Para simplificar, en todo el paquete suponemos que no hay valores
    # perdidos ni infinitos. np.isfinite es False tanto para NaN como para inf.
    if not np.isfinite(x).all():
        raise ValueError(f"'{arg_name}' contiene valores NaN o infinitos")

    return x.astype(float)


def _variance(x):
    """Varianza muestral de un vector numérico: la suma de los cuadrados de las
    diferencias con la media, dividida entre n - 1 (como var en R). Con un solo
    valor no se puede calcular (habría que dividir entre 0), así que devuelve
    NaN, igual que R."""

    if len(x) < 2:
        return np.nan

    return np.sum((x - np.mean(x)) ** 2) / (len(x) - 1)


def _toDataFrame(data):
    """Convierte los datos en un DataFrame (también acepta una matriz de numpy)
    y devuelve siempre una copia. Así las funciones nunca modifican los datos
    originales: en Python los DataFrames se pasan por referencia, de modo que
    si los modificáramos directamente cambiaríamos también los del usuario."""

    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)

    if not isinstance(data, pd.DataFrame):
        raise TypeError("Los datos deben ser un DataFrame, una matriz o un "
                        "objeto Dataset")

    return data.copy()


def _isNumeric(x):
    """Indica si una columna es numérica (entera o real). Las columnas
    booleanas no cuentan como numéricas (las tratamos como categóricas)."""
    return pd.api.types.is_numeric_dtype(x) and not pd.api.types.is_bool_dtype(x)


def _isCategory(x):
    """Indica si 'x' es una variable de tipo category de pandas (el equivalente
    a un factor de R), ya sea un pd.Categorical o una Series."""
    if isinstance(x, pd.Categorical):
        return True
    return isinstance(x, pd.Series) and x.dtype == "category"


def _isCategorical(x):
    """Indica si una columna se considera categórica: de tipo category o
    booleana. Las columnas de texto no se consideran categóricas."""
    return _isCategory(x) or pd.api.types.is_bool_dtype(x)


def _toBinaryClass(labels):
    """Convierte una variable clase binaria en un array de True / False.

    Si ya es booleana se devuelve tal cual (True = positivo). Si no, tiene que
    tener exactamente dos valores distintos y se considera positivo el
    segundo: en una variable categórica, la segunda categoría (en el orden de
    las categorías, como los niveles de un factor en R) y en otro caso el
    mayor de los dos valores (por ejemplo el 1 en una variable 0/1).
    """

    labels = pd.Series(labels)

    if pd.api.types.is_bool_dtype(labels):
        return labels.to_numpy(dtype=bool)

    if labels.dtype == "category":
        # Categorías en su orden, pero solo las que aparecen en los datos
        present = list(labels.unique())
        values = [category for category in labels.cat.categories
                  if category in present]
    else:
        values = sorted(labels.unique())

    if len(values) != 2:
        raise ValueError("La variable clase debe tener exactamente dos valores")

    return (labels == values[1]).to_numpy()


def _isBinaryClass(labels):
    """Indica si una variable clase tiene exactamente dos valores distintos,
    que es lo necesario para calcular el AUC."""
    return len(pd.Series(labels).unique()) == 2


def _checkBinaryClass(data, class_name):
    """Lanza un error si 'class_name' no es una variable clase binaria de
    'data'. La usan filterData (con metric="auc") y plotAUC."""

    if (class_name is None or class_name not in data.columns
            or not _isBinaryClass(data[class_name])):
        raise ValueError("Para calcular el AUC hay que indicar una variable "
                         "clase binaria ('class_name')")


def _transformNumericColumns(data, function):
    """Aplica una función (normalize o standardize) a todas las columnas
    numéricas de un DataFrame y deja el resto como están. La usan
    normalizeData y standardizeData para no repetir código."""

    numeric_columns = [column for column in data.columns
                       if _isNumeric(data[column])]

    if len(numeric_columns) == 0:
        warnings.warn("No hay variables numéricas: los datos no cambian")

    for column in numeric_columns:
        data[column] = function(data[column])

    return data


def _pairwiseMatrix(data, function):
    """Calcula una matriz aplicando una función a cada par de columnas de
    'data'. La usan la correlación y la información mutua.

    Entradas:
        data      DataFrame con las variables.
        function  Función de dos argumentos (x, y) que devuelve un número.

    Salida:
        DataFrame cuadrado con el resultado de cada par de variables.
    """

    variables = list(data.columns)
    n = len(variables)
    result = pd.DataFrame(np.zeros((n, n)), index=variables, columns=variables)

    # La medida es simétrica, así que solo calculamos la mitad de la matriz
    # (j >= i) y copiamos el valor en la posición simétrica.
    for i in range(n):
        for j in range(i, n):
            value = function(data.iloc[:, i], data.iloc[:, j])
            result.iloc[i, j] = value
            result.iloc[j, i] = value

    return result
