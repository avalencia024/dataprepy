# =============================================================================
# filtering.py
# -----------------------------------------------------------------------------
# Filtrado de variables según las métricas de computeMetrics: a partir de un
# dataset se obtiene otro con solo las variables que cumplen un requisito
# (por ejemplo, una entropía mayor que un umbral).
# =============================================================================

import numpy as np
import pandas as pd

from .dataset import Dataset
from .metrics import computeMetrics
from .utils import _checkBinaryClass, _toDataFrame


def filterData(data, metric, threshold, greater=True, keep_others=False,
               class_name=None):
    """Filtra las variables de un dataset según una de las métricas.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos.
    metric : str
        Métrica a usar: "variance", "auc" o "entropy".
    threshold : float
        Valor umbral (un número finito).
    greater : bool
        Si es True (por defecto) se quedan las variables con un valor de la
        métrica mayor que 'threshold'; si es False, las que lo tienen menor.
    keep_others : bool
        Qué hacer con las variables a las que no se aplica la métrica (por
        ejemplo las numéricas al filtrar por entropía). Si es False (por
        defecto) se eliminan, de modo que en el resultado solo quedan las
        variables que cumplen el requisito, además de la clase. Si es True se
        mantienen, lo que permite encadenar filtros sobre métricas distintas.
    class_name : str, opcional
        Variable clase. Hace falta para filtrar por AUC y tiene que ser
        binaria. Nunca se elimina. En un Dataset se usa su propia clase.

    Returns
    -------
    pandas.DataFrame o Dataset
        Los datos filtrados, del mismo tipo que 'data', con las variables en
        su orden original.

    Notes
    -----
    Las métricas se calculan con computeMetrics: la varianza y el AUC se
    aplican a las variables numéricas y la entropía a las categóricas. Para
    aplicar varios requisitos basta con llamar varias veces a la función; si
    son sobre métricas de tipos distintos (por ejemplo AUC y entropía), hay
    que usar keep_others=True para que el primer filtro no elimine las
    variables del otro tipo.

    Examples
    --------
    >>> datos = pd.DataFrame({"a": [1.0, 5.0, 9.0, 2.0],
    ...                       "b": [1.0, 1.1, 0.9, 1.0],
    ...                       "grupo": pd.Categorical(["x", "y", "x", "y"])})
    >>> list(dp.filterData(datos, metric="variance", threshold=1).columns)
    ['a']
    >>> list(dp.filterData(datos, metric="variance", threshold=1,
    ...                    keep_others=True).columns)
    ['a', 'grupo']
    """

    # Con un Dataset filtramos su DataFrame y creamos un Dataset nuevo
    if isinstance(data, Dataset):
        if class_name is None:
            class_name = data.class_name
        filtered = filterData(data.data, metric, threshold, greater,
                              keep_others, class_name)
        return Dataset(filtered, class_name=data.class_name, name=data.name)

    data = _toDataFrame(data)

    if metric not in ["variance", "auc", "entropy"]:
        raise ValueError("'metric' debe ser 'variance', 'auc' o 'entropy'")

    # El umbral tiene que ser un único número finito (con NaN, por ejemplo,
    # ninguna comparación se cumpliría y se eliminarían todas las variables)
    if (not isinstance(threshold, (int, float, np.integer, np.floating))
            or not np.isfinite(threshold)):
        raise ValueError("'threshold' debe ser un único valor numérico finito")

    # Para filtrar por AUC hace falta una clase binaria (si no, el AUC de
    # todas las variables sería NaN y no se filtraría nada)
    if metric == "auc":
        _checkBinaryClass(data, class_name)

    # Métricas de todas las variables y valores de la métrica elegida
    metrics = computeMetrics(data, class_name=class_name)
    values = metrics[metric]

    # Recorremos las variables y decidimos si se quedan o no
    selected = []

    for variable in metrics.index:
        value = values[variable]

        if pd.isna(value):
            # NaN: la métrica no se aplica a esta variable
            keep = keep_others
        elif greater:
            keep = value > threshold
        else:
            keep = value < threshold

        if keep:
            selected.append(variable)

    # Mantenemos el orden original de las columnas y siempre la clase
    columns = [column for column in data.columns
               if column in selected or column == class_name]

    return data[columns]
