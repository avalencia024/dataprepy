# =============================================================================
# plotting.py
# -----------------------------------------------------------------------------
# Visualización con matplotlib:
#   - plotROC:         curva ROC de una variable, con su AUC.
#   - plotAUC:         diagrama de barras con el AUC de cada variable numérica.
#   - plotCorrelation: matriz de correlaciones o de información mutua.
#   - plotEntropy:     diagrama de barras con la entropía normalizada.
#
# Las funciones crean una figura nueva y no llaman a plt.show(): en un
# notebook la figura se muestra sola y en un script hay que llamar a
# plt.show() después.
# =============================================================================

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .correlation import computeCorrelations
from .dataset import Dataset
from .metrics import calculateAUC, computeMetrics, entropy
from .utils import _checkBinaryClass, _isCategorical, _toDataFrame


def plotROC(data, title="Curva ROC"):
    """Dibuja la curva ROC de una variable numérica, con su AUC en el título.

    Parameters
    ----------
    data : pandas.DataFrame
        Tabla de dos columnas: la variable numérica y la clase (ver
        calculateAUC).
    title : str
        Título del gráfico (al final se le añade el AUC).

    Returns
    -------
    roc_points : pandas.DataFrame
        Puntos de la curva ROC.
    auc : float
        Área bajo la curva.

    Notes
    -----
    La diagonal discontinua es la curva de una variable que no separa las
    clases (AUC = 0.5).
    """

    roc_points, auc = calculateAUC(data)

    plt.figure(figsize=(5.5, 5))
    plt.plot(roc_points["FPR"], roc_points["TPR"], color="steelblue",
             marker="o", markersize=4, linewidth=2)

    # Diagonal de referencia (AUC = 0.5)
    plt.plot([0, 1], [0, 1], color="grey", linestyle="--")

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel("FPR (tasa de falsos positivos)")
    plt.ylabel("TPR (tasa de verdaderos positivos)")
    plt.title(f"{title} - AUC = {round(auc, 3)}")

    return roc_points, auc


def plotAUC(data, class_name=None, title="AUC de las variables numéricas"):
    """Diagrama de barras con el AUC de las variables numéricas, de mayor a
    menor.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos.
    class_name : str, opcional
        Variable clase (binaria). En un Dataset se usa su propia clase.
    title : str
        Título del gráfico.

    Returns
    -------
    pandas.Series
        AUC de cada variable numérica, de mayor a menor.

    Notes
    -----
    Las barras con AUC mayor o igual que 0.5 se dibujan en azul y las demás
    en rojo. La línea discontinua marca el 0.5.
    """

    if isinstance(data, Dataset):
        if class_name is None:
            class_name = data.class_name
        data = data.data

    data = _toDataFrame(data)
    _checkBinaryClass(data, class_name)

    # AUC de las variables numéricas (las demás tienen NaN)
    auc_values = computeMetrics(data, class_name)["auc"].dropna()
    auc_values = auc_values.sort_values(ascending=False)

    colors = []
    for value in auc_values:
        if value >= 0.5:
            colors.append("steelblue")
        else:
            colors.append("indianred")

    plt.figure(figsize=(7, 4.5))
    plt.bar(auc_values.index, auc_values, color=colors)
    plt.axhline(0.5, color="grey", linestyle="--")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.ylabel("AUC")
    plt.title(title)

    return auc_values


def plotCorrelation(data, kind="correlation", title=None):
    """Dibuja la matriz de correlaciones o de información mutua como un mapa de
    calor, con el valor escrito en cada casilla.

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos.
    kind : str
        "correlation" (por defecto) para la matriz de correlaciones entre las
        variables numéricas o "mutual_information" para la de información
        mutua entre las categóricas.
    title : str, opcional
        Título. Si no se indica se pone uno según el tipo de matriz.

    Returns
    -------
    pandas.DataFrame
        La matriz dibujada (calculada con computeCorrelations).

    Notes
    -----
    Las correlaciones van del azul (-1) al rojo (1) pasando por el blanco (0)
    y la información mutua del blanco (0) al verde (valor máximo).
    """

    correlation_matrix, mutual_information_matrix = computeCorrelations(data)

    if kind == "correlation":
        matrix = correlation_matrix
        colormap = "coolwarm"
        min_value, max_value = -1, 1
        default_title = "Matriz de correlaciones"
    elif kind == "mutual_information":
        matrix = mutual_information_matrix
        colormap = "Greens"
        min_value = 0
        max_value = None if matrix is None else matrix.max().max()
        default_title = "Matriz de información mutua"
    else:
        raise ValueError("'kind' debe ser 'correlation' o 'mutual_information'")

    if matrix is None or len(matrix) < 2:
        raise ValueError("Hacen falta al menos dos variables del tipo adecuado")

    if title is None:
        title = default_title

    n = len(matrix)
    size = max(5, 0.6 * n + 2)
    plt.figure(figsize=(size + 1, size))

    plt.imshow(matrix, cmap=colormap, vmin=min_value, vmax=max_value)
    plt.colorbar()

    # Nombres de las variables en los ejes
    plt.xticks(range(n), matrix.columns, rotation=90)
    plt.yticks(range(n), matrix.index)

    # Valor de cada casilla: la fila i, columna j está en x = j, y = i. En las
    # casillas más oscuras (valores cerca del máximo) escribimos en blanco
    # para que se pueda leer.
    for i in range(n):
        for j in range(n):
            value = matrix.iloc[i, j]
            if abs(value) > 0.7 * max_value:
                text_color = "white"
            else:
                text_color = "black"
            plt.text(j, i, round(value, 2), ha="center", va="center",
                     fontsize=8, color=text_color)

    plt.title(title)

    return matrix


def plotEntropy(data, title="Entropía normalizada"):
    """Diagrama de barras con la entropía normalizada de cada variable
    categórica (por ejemplo, de un dataset discretizado).

    Parameters
    ----------
    data : pandas.DataFrame o Dataset
        Datos. Las variables que no son categóricas se ignoran.
    title : str
        Título del gráfico.

    Returns
    -------
    pandas.Series
        Entropía normalizada de cada variable categórica.

    Notes
    -----
    La entropía se divide entre su valor máximo, log2 del número de
    categorías, para que todas las variables queden entre 0 y 1 y se puedan
    comparar aunque tengan distinto número de valores.
    """

    if isinstance(data, Dataset):
        data = data.data

    data = _toDataFrame(data)

    normalized_entropy = {}

    for column in data.columns:
        if _isCategorical(data[column]):
            x = pd.Categorical(data[column])
            num_values = len(x.categories)

            # Con una sola categoría la entropía máxima es 0 y no se puede
            # dividir: la entropía normalizada es 0
            if num_values <= 1:
                normalized_entropy[column] = 0.0
            else:
                normalized_entropy[column] = entropy(x) / np.log2(num_values)

    if len(normalized_entropy) == 0:
        raise ValueError("No hay variables categóricas: discretiza antes los "
                         "datos con discretizeData")

    normalized_entropy = pd.Series(normalized_entropy)

    plt.figure(figsize=(7, 4.5))
    plt.bar(normalized_entropy.index, normalized_entropy, color="darkseagreen")
    plt.ylim(0, 1)
    plt.xticks(rotation=90)
    plt.ylabel("Entropía normalizada")
    plt.title(title)

    return normalized_entropy
