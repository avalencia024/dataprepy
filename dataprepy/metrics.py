# =============================================================================
# metrics.py
# -----------------------------------------------------------------------------
# Métricas de las variables de un dataset:
#   - entropy:        entropía de una variable categórica.
#   - calculateAUC:   curva ROC y AUC de una variable numérica respecto a una
#                     clase binaria.
#   - computeMetrics: varianza y AUC de las variables numéricas y entropía de
#                     las categóricas, según el tipo de cada variable.
# =============================================================================

import warnings

import numpy as np
import pandas as pd

from .dataset import Dataset
from .utils import (_checkNumeric, _isBinaryClass, _isCategorical, _isCategory,
                    _isNumeric, _toBinaryClass, _toDataFrame, _variance)


def entropy(x):
    """Entropía (en bits) de una variable categórica.

    Parameters
    ----------
    x : pandas.Categorical o pandas.Series de tipo category
        Valores de la variable.

    Returns
    -------
    float
        Entropía. Vale 0 si la variable siempre toma el mismo valor y como
        máximo log2(k), siendo k el número de categorías.

    Notes
    -----
    Primero se estima la probabilidad de cada posible valor (cada categoría,
    aparezca o no en los datos) como el número de apariciones dividido entre
    el total de valores. Después se aplica H(X) = sum(-p_i * log2(p_i)), sin
    contar los valores con probabilidad 0.

    Por ejemplo, con los valores posibles a, b y c y los datos
    (a, a, c, c, c), las probabilidades son 0.4, 0 y 0.6 y la entropía es
    -0.4 * log2(0.4) - 0.6 * log2(0.6) = 0.971.

    Examples
    --------
    >>> x = pd.Categorical(["a", "a", "c", "c", "c"], categories=["a", "b", "c"])
    >>> round(dp.entropy(x), 3)
    0.971
    """

    if not _isCategory(x):
        raise TypeError("'x' debe ser una variable categórica (pd.Categorical)")

    x = pd.Categorical(x)

    # Como en el resto del paquete, no se admiten valores perdidos (contarían
    # en el total pero en ninguna categoría)
    if x.isna().any():
        raise ValueError("'x' contiene valores perdidos (NaN)")

    # Probabilidad de cada categoría: número de veces que aparece entre el
    # número total de valores
    probabilities = []
    for category in x.categories:
        probabilities.append(np.sum(x == category) / len(x))
    probabilities = np.array(probabilities)

    # Las probabilidades 0 no suman nada (y log2(0) daría -inf)
    probabilities = probabilities[probabilities > 0]

    return float(-np.sum(probabilities * np.log2(probabilities)))


def calculateAUC(data):
    """Curva ROC y área bajo la curva (AUC) de una variable numérica.

    Mide lo bien que una variable numérica, por sí sola, separa las dos
    clases de una variable binaria.

    Parameters
    ----------
    data : pandas.DataFrame
        Tabla de dos columnas: la primera es la variable numérica y la segunda
        la clase. La clase puede ser booleana (True / False) o tener dos
        valores distintos; en ese caso el positivo es la segunda categoría si
        es una variable categórica (como en R) y, si no, el mayor de los dos
        valores (por ejemplo el 1 en una variable 0/1).

    Returns
    -------
    roc_points : pandas.DataFrame
        Puntos de la curva ROC, con las columnas FPR y TPR.
    auc : float
        Área bajo la curva ROC.

    Notes
    -----
    Para cada punto de corte, las filas con valor mayor o igual que el corte
    se predicen como positivas y el resto como negativas. Comparando con la
    clase real se cuentan los TP, FP, TN y FN y se calculan
    TPR = TP / (TP + FN) y FPR = FP / (FP + TN). Solo cambian el TPR y el FPR
    los cortes que coinciden con los valores de los datos, así que solo se
    usan esos (de mayor a menor). La curva empieza en (0, 0) y el área se
    calcula sumando los trapecios entre puntos consecutivos (regla del
    trapecio).

    Un AUC de 1 indica que, al ordenar por la variable, las dos clases quedan
    perfectamente separadas; un AUC menor que 0.5, que los valores bajos de la
    variable se asocian a la clase positiva; y un AUC de 0.5, que ordenar por
    la variable no separa las clases. Un AUC cercano a 0.5 no quiere decir
    necesariamente que la variable no tenga relación con la clase: no detecta,
    por ejemplo, que los positivos estén en los dos extremos de la variable.

    Examples
    --------
    >>> datos = pd.DataFrame({"valor": [0.1, 0.4, 0.35, 0.8],
    ...                       "clase": [False, False, True, True]})
    >>> roc_points, auc = dp.calculateAUC(datos)
    >>> auc
    0.75
    """

    data = _toDataFrame(data)

    if data.shape[1] != 2:
        raise ValueError("'data' debe tener dos columnas: la variable numérica "
                         "y la clase")

    # Separamos la variable y la clase (True = positivo). Como en el resto del
    # paquete, la clase no puede tener valores perdidos ni, si es numérica,
    # infinitos. Lo comprobamos antes de pasarla a True / False.
    values = _checkNumeric(data.iloc[:, 0], "primera columna de data")
    classes = data.iloc[:, 1]

    if classes.isna().any() or (_isNumeric(classes) and np.isinf(classes).any()):
        raise ValueError("La clase contiene valores perdidos (NaN) o infinitos")

    labels = _toBinaryClass(classes)

    if labels.all() or not labels.any():
        raise ValueError("La clase debe tener ejemplos positivos y negativos")

    # Puntos de corte: los valores distintos de la variable, de mayor a menor
    # (np.unique los ordena de menor a mayor y [::-1] les da la vuelta)
    cut_points = np.unique(values)[::-1]

    # La curva ROC empieza siempre en (0, 0)
    fpr = [0.0]
    tpr = [0.0]

    for cut_point in cut_points:

        # Predecimos como positivas las filas con valor >= punto de corte
        prediction = values >= cut_point

        # Matriz de confusión (& es el "y" elemento a elemento y ~ la negación)
        tp = np.sum(prediction & labels)
        fp = np.sum(prediction & ~labels)
        tn = np.sum(~prediction & ~labels)
        fn = np.sum(~prediction & labels)

        fpr.append(fp / (fp + tn))
        tpr.append(tp / (tp + fn))

    roc_points = pd.DataFrame({"FPR": fpr, "TPR": tpr})

    # Área: suma de los trapecios entre puntos consecutivos, anchura
    # (diferencia de FPR) por altura media (media de los dos TPR)
    fpr = np.array(fpr)
    tpr = np.array(tpr)
    auc = np.sum(np.diff(fpr) * (tpr[1:] + tpr[:-1]) / 2)

    return roc_points, float(auc)


def computeMetrics(data, class_name=None):
    """Métricas de las variables de un dataset según su tipo.

    Calcula la varianza y el AUC de las variables numéricas y la entropía de
    las categóricas. La función reconoce el tipo de cada variable.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos.
    class_name : str, opcional
        Variable clase (binaria) con la que se calcula el AUC. Si no se indica
        no se calcula el AUC. En un Dataset se usa su propia variable clase.

    Returns
    -------
    pandas.DataFrame
        Una fila por variable (menos la clase) con las columnas "type"
        ("numeric", "categorical" u otro), "variance", "auc" y "entropy". Las
        métricas que no corresponden a una variable valen NaN.

    Notes
    -----
    Son numéricas las variables enteras y reales, y categóricas las de tipo
    category y las booleanas. A las de texto no se les calcula nada. La
    varianza es la muestral (divide entre n - 1, como var en R). Si la clase
    no es binaria (por ejemplo Species en iris) se avisa con un warning y el
    AUC queda como NaN, pero el resto de métricas se calculan igual.

    Como el resto del paquete, la función no admite valores perdidos ni
    infinitos en las variables numéricas y categóricas ni en la clase (da un
    error). Así, un NaN en el resultado indica que la métrica no se aplica a
    esa variable, salvo en un caso: con una sola observación la varianza
    muestral no está definida (necesita al menos dos valores) y también vale
    NaN.

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [0.1, 0.4, 0.35, 0.8],
    ...                       "grupo": pd.Categorical(["a", "a", "b", "b"]),
    ...                       "clase": [0, 0, 1, 1]})
    >>> metricas = dp.computeMetrics(datos, class_name="clase")
    >>> metricas["type"].tolist()
    ['numeric', 'categorical']
    """

    # En un Dataset, si no se indica otra cosa, usamos su variable clase
    if isinstance(data, Dataset):
        if class_name is None:
            class_name = data.class_name
        data = data.data

    data = _toDataFrame(data)

    # El AUC solo se calcula si hay una clase binaria
    compute_auc = False

    if class_name is not None:
        if class_name not in data.columns:
            raise ValueError("'class_name' debe ser una de las columnas")

        if _isBinaryClass(data[class_name]):
            compute_auc = True
        else:
            warnings.warn(f"La variable clase '{class_name}' no es binaria: "
                          f"no se calcula el AUC")

    # Para cada variable guardamos una fila con sus métricas
    rows = []
    names = []

    for variable in data.columns:

        if variable == class_name:
            continue

        x = data[variable]

        # Por defecto las métricas no se aplican (NaN)
        variable_type = str(x.dtype)
        variance = np.nan
        auc = np.nan
        entropy_value = np.nan

        if _isNumeric(x):
            variable_type = "numeric"
            # _checkNumeric comprueba que no haya valores NaN ni infinitos
            variance = _variance(_checkNumeric(x, variable))
            if compute_auc:
                auc = calculateAUC(data[[variable, class_name]])[1]

        elif _isCategorical(x):
            variable_type = "categorical"
            # pd.Categorical convierte también las booleanas en categóricas
            entropy_value = entropy(pd.Categorical(x))

        rows.append([variable_type, variance, auc, entropy_value])
        names.append(variable)

    # Tabla con una fila por variable (el nombre de la variable es el índice)
    metrics = pd.DataFrame(rows, index=names,
                           columns=["type", "variance", "auc", "entropy"])

    return metrics
