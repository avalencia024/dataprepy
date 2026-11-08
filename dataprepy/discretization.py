# =============================================================================
# discretization.py
# -----------------------------------------------------------------------------
# Discretización por igual anchura e igual frecuencia:
#   - discretize:     discretiza con unos puntos de corte dados.
#   - discretizeEW:   igual anchura (equal width) para un vector.
#   - discretizeEF:   igual frecuencia (equal frequency) para un vector.
#   - discretizeData: cualquiera de los dos métodos para un dataset completo.
#
# Los intervalos son cerrados por la derecha, (a, b], y el primero y el último
# llegan a -inf e inf para poder discretizar valores nuevos que caigan fuera
# del rango de los datos originales. El resultado es un pd.Categorical, que es
# el equivalente en Python a un factor de R.
# =============================================================================

import warnings

import numpy as np
import pandas as pd

from .dataset import Dataset
from .utils import _checkNumeric, _isNumeric, _toDataFrame


def discretize(x, cut_points):
    """Discretiza un vector numérico con unos puntos de corte dados.

    Parameters
    ----------
    x : list, numpy.ndarray o pandas.Series
        Vector numérico a discretizar.
    cut_points : list o numpy.ndarray
        Puntos de corte. No hace falta que estén ordenados. Puede estar vacío:
        en ese caso hay un único intervalo, (-inf, inf).

    Returns
    -------
    pandas.Categorical
        Un valor por cada valor de 'x', con el intervalo en el que cae. Las
        categorías son todos los intervalos en orden (aunque alguno quede
        vacío) y su nombre indica los límites, por ejemplo "(5.5, 10.5]".

    Notes
    -----
    Con k puntos de corte salen k + 1 intervalos: (-inf, c1], (c1, c2], ...,
    (ck, inf). Un valor igual a un punto de corte pertenece al intervalo de su
    izquierda. Sirve para aplicar a valores nuevos los puntos de corte que
    devuelven discretizeEW y discretizeEF.

    Para que se lean bien, en los nombres de los intervalos los límites se
    escriben con 4 cifras significativas (como en R); los puntos de corte
    exactos son los que devuelven discretizeEW y discretizeEF.

    Examples
    --------
    >>> x = [11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4]
    >>> list(dp.discretize(x, cut_points=[5.5, 10.5, 15.5]))
    ['(10.5, 15.5]', '(5.5, 10.5]', '(-inf, 5.5]', '(-inf, 5.5]', '(-inf, 5.5]', '(15.5, inf)', '(5.5, 10.5]']
    """

    x = _checkNumeric(x, "x")

    # Los puntos de corte pueden estar vacíos (un único intervalo), pero si hay
    # alguno tiene que ser un número finito
    cut_points = np.array(cut_points, dtype=float)
    if not np.isfinite(cut_points).all():
        raise ValueError("'cut_points' no puede tener valores NaN ni infinitos")

    # Ordenamos los puntos de corte y quitamos los repetidos
    cut_points = np.unique(cut_points)

    # Los límites de los intervalos van de -inf a inf pasando por los puntos
    # de corte. Para los nombres los escribimos con 4 cifras significativas,
    # y si al redondear dos límites quedan iguales, los escribimos enteros
    # (si no, dos intervalos tendrían el mismo nombre).
    limits = np.concatenate(([-np.inf], cut_points, [np.inf]))
    texts = [f"{limit:.4g}" for limit in limits]
    if len(set(texts)) < len(texts):
        texts = [f"{limit}" for limit in limits]

    interval_names = []
    for i in range(len(limits) - 1):
        interval_names.append(f"({texts[i]}, {texts[i + 1]}]")

    # El último intervalo llega a infinito, así que es abierto
    interval_names[-1] = f"({texts[-2]}, inf)"

    # El intervalo de cada valor es el número de puntos de corte que quedan
    # por debajo de él. Usamos ">" para que un valor igual a un punto de corte
    # quede en el intervalo de la izquierda.
    result = []
    for value in x:
        interval = np.sum(value > cut_points)
        result.append(interval_names[interval])

    # Al indicar las categorías aparecen todos los intervalos, aunque alguno
    # no tenga ningún valor
    return pd.Categorical(result, categories=interval_names)


def discretizeEW(x, num_bins):
    """Discretización por igual anchura (equal width).

    Divide el rango de los valores en 'num_bins' intervalos de la misma
    anchura.

    Parameters
    ----------
    x : list, numpy.ndarray o pandas.Series
        Vector numérico a discretizar.
    num_bins : int
        Número de intervalos (2 o más).

    Returns
    -------
    x_discretized : pandas.Categorical
        Valores discretizados (ver discretize).
    cut_points : numpy.ndarray
        Puntos de corte, para poder discretizar más valores con discretize.

    Notes
    -----
    La anchura de los intervalos es (max(x) - min(x)) / num_bins y los
    num_bins - 1 puntos de corte se ponen a esa distancia empezando en el
    mínimo. Por ejemplo, con (11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4) y 4
    intervalos, el rango va de 0.5 a 20.5, la anchura es 5 y los puntos de
    corte son 5.5, 10.5 y 15.5. Si todos los valores son iguales la anchura es
    0 y no se puede dividir el rango: no hay puntos de corte y todos los
    valores quedan en un único intervalo (se avisa con un warning).

    Examples
    --------
    >>> x = [11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4]
    >>> x_discretized, cut_points = dp.discretizeEW(x, num_bins=4)
    >>> cut_points
    array([ 5.5, 10.5, 15.5])
    """

    x = _checkNumeric(x, "x")
    _checkNumBins(num_bins)

    min_value = np.min(x)
    max_value = np.max(x)

    # Anchura común de todos los intervalos
    bin_width = (max_value - min_value) / num_bins

    if bin_width == 0:
        # Variable constante: no se puede dividir el rango, así que no hay
        # puntos de corte y todos los valores quedan en un único intervalo
        warnings.warn("Todos los valores son iguales: se crea un único intervalo")
        cut_points = np.array([])
    else:
        # num_bins intervalos necesitan num_bins - 1 puntos de corte: min + 1 *
        # anchura, min + 2 * anchura, ..., min + (num_bins - 1) * anchura
        cut_points = min_value + bin_width * np.arange(1, num_bins)

    x_discretized = discretize(x, cut_points)

    return x_discretized, cut_points


def discretizeEF(x, num_bins):
    """Discretización por igual frecuencia (equal frequency).

    Busca los puntos de corte que hacen que en cada intervalo caiga el mismo
    número de valores (más o menos uno).

    Parameters
    ----------
    x : list, numpy.ndarray o pandas.Series
        Vector numérico a discretizar.
    num_bins : int
        Número de intervalos (entre 2 y el número de valores de 'x').

    Returns
    -------
    x_discretized : pandas.Categorical
        Valores discretizados (ver discretize).
    cut_points : numpy.ndarray
        Puntos de corte, para poder discretizar más valores con discretize.

    Notes
    -----
    Se ordenan los valores y el intervalo i termina en la posición
    round(i * n / num_bins), siendo n el número de valores. El punto de corte
    se pone en el punto medio entre el último valor de un intervalo y el
    primero del siguiente.

    Empates: los valores repetidos siempre quedan en el mismo intervalo,
    porque un punto de corte solo se puede poner entre dos valores distintos.
    Si la posición calculada cae entre dos valores iguales, el corte se mueve
    al hueco entre valores distintos más cercano a la posición ideal
    (i * n / num_bins). Por eso, con valores repetidos las frecuencias pueden
    quedar algo descompensadas, y si dos cortes acaban en el mismo hueco salen
    menos intervalos de los pedidos (se avisa con un warning). Nunca queda
    ningún intervalo vacío. Si todos los valores son iguales se crea un único
    intervalo.

    Examples
    --------
    >>> x = [11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4]
    >>> x_discretized, cut_points = dp.discretizeEF(x, num_bins=4)
    >>> np.round(cut_points, 2)
    array([ 3.25,  9.3 , 10.85])
    """

    x = _checkNumeric(x, "x")
    _checkNumBins(num_bins)

    n = len(x)

    if num_bins > n:
        raise ValueError("'num_bins' no puede ser mayor que el número de valores")

    x_sorted = np.sort(x)

    # Posición ideal (contando desde 1) del último valor de cada intervalo,
    # menos el último intervalo, que llega hasta el final. Como las posiciones
    # tienen que ser números enteros, las redondeamos (np.round redondea los .5
    # al par más cercano, igual que round en R).
    ideal_positions = np.arange(1, num_bins) * n / num_bins
    positions = np.round(ideal_positions).astype(int)

    # Un punto de corte solo se puede poner entre dos valores distintos (si no,
    # separaría valores iguales). 'gaps' son las posiciones p en las que el
    # valor p y el siguiente son distintos. Como en Python los índices empiezan
    # en 0, el valor de la posición p es x_sorted[p - 1] y el siguiente es
    # x_sorted[p].
    gaps = [p for p in range(1, n) if x_sorted[p - 1] != x_sorted[p]]

    if len(gaps) == 0:
        # Variable constante: no se puede separar ningún valor
        warnings.warn("Todos los valores son iguales: se crea un único intervalo")
        cut_points = np.array([])

    else:
        # Si en una posición hay dos valores iguales, movemos el corte al hueco
        # entre valores distintos más cercano a la posición ideal
        for i in range(len(positions)):
            if positions[i] not in gaps:
                distances = [abs(gap - ideal_positions[i]) for gap in gaps]
                positions[i] = gaps[np.argmin(distances)]

        # Dos cortes pueden acabar en el mismo hueco. En ese caso nos quedamos
        # con uno solo (habrá menos intervalos de los pedidos).
        if len(np.unique(positions)) < len(positions):
            warnings.warn("Por los valores repetidos salen menos intervalos de "
                          "los pedidos")
            positions = np.unique(positions)

        # Cada punto de corte va a mitad de camino entre el último valor de un
        # intervalo y el primero del siguiente
        cut_points = (x_sorted[positions - 1] + x_sorted[positions]) / 2

    x_discretized = discretize(x, cut_points)

    return x_discretized, cut_points


def discretizeData(data, num_bins, method="EW"):
    """Discretiza todas las variables numéricas de un dataset.

    Aplica discretizeEW o discretizeEF a cada variable numérica. Las demás
    variables se dejan como están.

    Parameters
    ----------
    data : pandas.DataFrame, numpy.ndarray o Dataset
        Datos a discretizar. Si es un Dataset, la variable clase no se
        discretiza aunque sea numérica.
    num_bins : int
        Número de intervalos para cada variable.
    method : str
        "EW" (igual anchura, valor por defecto) o "EF" (igual frecuencia).

    Returns
    -------
    data_discretized : pandas.DataFrame o Dataset
        Los datos discretizados, del mismo tipo que 'data'.
    cut_points : dict
        Puntos de corte de cada variable discretizada (la clave es el nombre
        de la variable).

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [1.0, 2.0, 3.0, 4.0],
    ...                       "grupo": ["a", "a", "b", "b"]})
    >>> datos_disc, cut_points = dp.discretizeData(datos, num_bins=2)
    >>> list(datos_disc["x"])
    ['(-inf, 2.5]', '(-inf, 2.5]', '(2.5, inf)', '(2.5, inf)']
    """

    # Con un Dataset discretizamos solo las variables que no son la clase y
    # las volvemos a meter en un nuevo Dataset
    if isinstance(data, Dataset):
        predictors = data.data[data.getPredictorNames()]
        predictors, cut_points = discretizeData(predictors, num_bins, method)
        return data._replacePredictors(predictors), cut_points

    data = _toDataFrame(data)

    if method == "EW":
        discretize_function = discretizeEW
    elif method == "EF":
        discretize_function = discretizeEF
    else:
        raise ValueError("'method' debe ser 'EW' o 'EF'")

    # Discretizamos cada columna numérica y guardamos sus puntos de corte
    cut_points = {}

    for column in data.columns:
        if _isNumeric(data[column]):
            data[column], cut_points[column] = discretize_function(data[column],
                                                                   num_bins)

    return data, cut_points


def _checkNumBins(num_bins):
    """Comprueba que el número de intervalos sea un entero mayor o igual que 2."""

    if not isinstance(num_bins, int) or num_bins < 2:
        raise ValueError("'num_bins' debe ser un número entero mayor o igual "
                         "que 2")
