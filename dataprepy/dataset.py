# =============================================================================
# dataset.py
# -----------------------------------------------------------------------------
# Gestión de datasets: clase Dataset, que agrupa los datos con el nombre de su
# variable clase y un nombre descriptivo. Tiene tres atributos:
#   - data:       DataFrame con todas las variables (incluida la clase).
#   - class_name: nombre de la variable clase o None si no hay.
#   - name:       nombre descriptivo del dataset o None.
#
# El tipo de cada variable es el tipo de su columna en 'data': entera o real
# para las numéricas, category o booleana para las categóricas y texto para
# las demás (a las que no se les calcula ninguna métrica).
#
# Las funciones de preprocesado del paquete aceptan tanto un DataFrame como
# un Dataset. Cuando reciben un Dataset (lo comprueban con isinstance) saben
# cuál es la variable clase, de modo que no la transforman (no la normalizan,
# no la discretizan...) y la usan automáticamente para calcular el AUC.
# =============================================================================

import pandas as pd

from .utils import _toDataFrame


class Dataset:
    """Dataset con su variable clase y un nombre descriptivo.

    Parameters
    ----------
    data : pandas.DataFrame
        Datos. Cada columna es una variable (también vale una matriz de numpy).
    class_name : str, opcional
        Nombre de la columna que es la variable clase. Por defecto None
        (dataset no supervisado).
    name : str, opcional
        Nombre descriptivo del dataset. Por defecto None.

    Notes
    -----
    Los tres argumentos se guardan en los atributos data, class_name y name.
    Los datos se copian, así que los cambios posteriores en el DataFrame
    original no afectan al objeto.

    El tipo de cada variable es el tipo de su columna: numérica (entera o
    real), categórica (category o booleana) o texto.

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [1.5, 2.0, 0.5], "grupo": ["a", "b", "a"]})
    >>> ds = dp.Dataset(datos, class_name="grupo", name="Ejemplo")
    >>> ds.class_name
    'grupo'
    """

    def __init__(self, data, class_name=None, name=None):

        data = _toDataFrame(data)

        # La variable clase, si se indica, tiene que ser una de las columnas
        if class_name is not None and class_name not in data.columns:
            raise ValueError("'class_name' debe ser una de las columnas de 'data'")

        self.data = data
        self.class_name = class_name
        self.name = name

    def __str__(self):
        """Texto que se muestra al hacer print del objeto: nombre, dimensiones,
        variable clase, tipo de cada variable y primeras filas."""

        text = "Objeto de clase 'Dataset'\n"

        if self.name is not None:
            text += f"Nombre: {self.name}\n"

        text += f"Dimensiones: {self.data.shape[0]} filas x {self.data.shape[1]} variables\n"

        if self.class_name is None:
            text += "Variable clase: ninguna (dataset no supervisado)\n"
        else:
            text += f"Variable clase: {self.class_name}\n"

        text += "\nTipos de las variables:\n"
        text += self.data.dtypes.to_string() + "\n"
        text += "\nPrimeras filas:\n"
        text += self.data.head().to_string()

        return text

    def summary(self):
        """Resumen del dataset: devuelve la tabla de métricas de todas sus
        variables (la de computeMetrics), usando su variable clase para el AUC.
        """

        # Importamos computeMetrics aquí y no al principio del fichero porque
        # metrics.py importa a su vez este fichero (importación circular)
        from .metrics import computeMetrics

        return computeMetrics(self)

    def getPredictorNames(self):
        """Devuelve la lista de variables que no son la clase."""
        return [column for column in self.data.columns
                if column != self.class_name]

    def _replacePredictors(self, new_predictors):
        """Devuelve un nuevo Dataset con las variables predictoras cambiadas por
        las de 'new_predictors' (un DataFrame con las mismas columnas). La
        clase, el nombre y el orden de las columnas se mantienen. Lo usan las
        funciones de preprocesado cuando reciben un Dataset."""

        new_data = new_predictors.copy()

        if self.class_name is not None:
            new_data[self.class_name] = self.data[self.class_name]

        # Dejamos las columnas en el orden original
        new_data = new_data[list(self.data.columns)]

        return Dataset(new_data, class_name=self.class_name, name=self.name)


def isDataset(x):
    """Comprueba si un objeto es un Dataset válido.

    Parameters
    ----------
    x : object
        Objeto a comprobar.

    Returns
    -------
    bool
        True si 'x' es de la clase Dataset, sus datos son un DataFrame y la
        variable clase (si la hay) es una de sus columnas. False si no.

    Examples
    --------
    >>> datos = pd.DataFrame({"x": [1, 2, 3], "y": [0, 1, 0]})
    >>> dp.isDataset(dp.Dataset(datos, class_name="y"))
    True
    >>> dp.isDataset(datos)
    False
    """

    if not isinstance(x, Dataset):
        return False

    # Los atributos se pueden cambiar a mano después de crear el objeto, así
    # que comprobamos también que el contenido tenga sentido
    if not isinstance(x.data, pd.DataFrame):
        return False

    return x.class_name is None or x.class_name in x.data.columns
