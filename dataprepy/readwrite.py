# =============================================================================
# readwrite.py
# -----------------------------------------------------------------------------
# Gestión de datasets: lectura y escritura.
#   - readCSV:      lee un CSV y asigna a cada columna el tipo adecuado.
#   - writeDataset: guarda un Dataset en un formato propio de texto.
#   - readDataset:  lee los ficheros guardados con writeDataset.
#
# El formato propio (extensión .dts) tiene unas líneas de cabecera que empiezan
# por '#' y después los datos en CSV. Es el mismo formato que el del paquete de
# R, así que los ficheros se pueden leer desde los dos lenguajes. Por eso los
# tipos se escriben con los nombres de R (numeric, integer, factor, logical y
# character). Ejemplo:
#
#   #Variables: 3
#   #Name: Ejemplo
#   #Class: grupo
#   #Types: numeric,integer,factor
#   #Levels: grupo=A|B
#   "altura","edad","grupo"
#   1.7,20,"A"
#   ...
# =============================================================================

import pandas as pd

from .dataset import Dataset
from .utils import _isNumeric


def readCSV(file_path, max_categories=5, sep=",", dec="."):
    """Lee un fichero CSV y asigna a cada columna el tipo adecuado.

    Parameters
    ----------
    file_path : str
        Ruta del fichero CSV.
    max_categories : int
        Una columna con menos de 'max_categories' valores distintos se
        considera categórica. Por defecto 5.
    sep : str
        Separador de columnas. Por defecto ",".
    dec : str
        Separador de decimales. Por defecto ".".

    Returns
    -------
    pandas.DataFrame
        Los datos con el tipo de cada columna ya decidido.

    Notes
    -----
    El fichero se lee con pd.read_csv y después se decide el tipo de cada
    columna en este orden: si tiene pocos valores distintos, category (con
    las categorías como texto, igual que los niveles de un factor de R); si
    es numérica y todos sus valores son enteros, int; si es numérica, float;
    y si no, se deja como texto.
    """

    data = pd.read_csv(file_path, sep=sep, decimal=dec)

    for column in data.columns:
        x = data[column]

        if len(x.unique()) < max_categories:
            # Categórica. Ordenamos las categorías por su valor original (para
            # que, por ejemplo, 10 vaya detrás de 4) y las pasamos a texto. Los
            # valores perdidos no son una categoría (dropna los quita).
            categories = [str(value) for value in sorted(x.dropna().unique())]
            data[column] = pd.Categorical(x.astype(str), categories=categories)

        elif _isNumeric(x) and (x == x.round()).all():
            # Todos los valores son enteros. Ponemos int64 y no int porque en
            # Windows, con numpy anterior a la 2, int es int32.
            data[column] = x.astype("int64")

        elif _isNumeric(x):
            data[column] = x.astype(float)

    return data


def writeDataset(data, file_path):
    """Guarda un Dataset (o un DataFrame) en un fichero con formato propio.

    A diferencia de un CSV normal, el fichero guarda también el tipo de cada
    variable, las categorías de las variables categóricas, la variable clase
    y el nombre del dataset.

    Parameters
    ----------
    data : Dataset o pandas.DataFrame
        Datos a guardar.
    file_path : str
        Ruta del fichero (se recomienda la extensión .dts).

    Returns
    -------
    None
        No devuelve nada: solo escribe el fichero.

    Notes
    -----
    La cabecera tiene estas líneas (solo la primera y #Types son
    obligatorias):

    - #Variables: número de variables.
    - #Name: nombre del dataset.
    - #Class: variable clase.
    - #Types: tipo de cada variable (numeric, integer, factor, logical o
      character).
    - #Levels: una línea por cada variable categórica, con sus categorías en
      orden separadas por "|", por ejemplo "#Levels: grupo=A|B".

    Después van los datos en CSV. Como el resto del paquete, el formato no
    admite valores perdidos: si los datos tienen alguno se da un error.
    """

    if not isinstance(data, Dataset):
        data = Dataset(data)

    df = data.data.copy()

    # No se admiten valores perdidos. Así, al leer el fichero, un texto como
    # "NA" nunca se confunde con un valor perdido.
    if df.isna().any().any():
        raise ValueError("Los datos tienen valores perdidos (NaN): no se pueden "
                         "guardar en este formato")

    # Tipo de cada columna con los nombres de R
    types = []
    for column in df.columns:
        x = df[column]
        if x.dtype == "category":
            types.append("factor")
        elif pd.api.types.is_bool_dtype(x):
            types.append("logical")
            # Los valores lógicos se escriben como en R
            df[column] = x.map({True: "TRUE", False: "FALSE"})
        elif pd.api.types.is_integer_dtype(x):
            types.append("integer")
        elif pd.api.types.is_float_dtype(x):
            types.append("numeric")
        else:
            types.append("character")

    # Primero escribimos la cabecera ('with' cierra el fichero al terminar)
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(f"#Variables: {df.shape[1]}\n")

        if data.name is not None:
            file.write(f"#Name: {data.name}\n")

        if data.class_name is not None:
            file.write(f"#Class: {data.class_name}\n")

        file.write("#Types: " + ",".join(types) + "\n")

        for column, type_name in zip(df.columns, types):
            if type_name == "factor":
                levels = [str(level) for level in df[column].cat.categories]
                file.write(f"#Levels: {column}=" + "|".join(levels) + "\n")

    # Después añadimos los datos al final del fichero (mode="a")
    df.to_csv(file_path, mode="a", index=False, encoding="utf-8")


def readDataset(file_path):
    """Lee un fichero guardado con writeDataset (o con writeDataset del
    paquete de R) y devuelve un Dataset con los mismos tipos, categorías,
    variable clase y nombre que el original.

    Parameters
    ----------
    file_path : str
        Ruta del fichero.

    Returns
    -------
    Dataset
        El dataset leído.

    Notes
    -----
    Se comprueba que la primera línea sea "#Variables: n", que exista la línea
    "#Types:" y que el número de columnas de los datos sea n.
    """

    # Leemos las líneas de la cabecera (las primeras que empiezan por '#')
    header = []
    with open(file_path, encoding="utf-8") as file:
        for line in file:
            if not line.startswith("#"):
                break
            header.append(line.strip())

    if len(header) == 0 or not header[0].startswith("#Variables: "):
        raise ValueError("La primera línea tiene que tener el formato:\n"
                         "#Variables: numero_de_variables")

    num_variables = int(header[0].split(": ")[1])

    # Información de la cabecera
    name = None
    class_name = None
    types = None
    levels = {}

    for line in header[1:]:
        key, value = line.split(": ", 1)
        if key == "#Name":
            name = value
        elif key == "#Class":
            class_name = value
        elif key == "#Types":
            types = value.split(",")
        elif key == "#Levels":
            column, column_levels = value.split("=", 1)
            levels[column] = column_levels.split("|")

    if types is None:
        raise ValueError("Falta la línea '#Types:' en la cabecera")

    # Leemos los datos como texto (saltando la cabecera) y después pasamos
    # cada columna a su tipo. Por defecto read_csv convierte textos como "NA",
    # "N/A" o "" en valores perdidos; con keep_default_na=False no convierte
    # nada (el fichero no tiene valores perdidos, writeDataset no los admite).
    data = pd.read_csv(file_path, skiprows=len(header), dtype=str,
                       keep_default_na=False, encoding="utf-8")

    if data.shape[1] != num_variables or len(types) != num_variables:
        raise ValueError("El número de variables no coincide con la cabecera")

    for column, type_name in zip(data.columns, types):
        if type_name == "numeric":
            data[column] = data[column].astype(float)
        elif type_name == "integer":
            data[column] = data[column].astype("int64")
        elif type_name == "logical":
            data[column] = data[column] == "TRUE"
        elif type_name == "factor":
            data[column] = pd.Categorical(data[column], categories=levels[column])
        elif type_name != "character":
            raise ValueError(f"Tipo de variable desconocido: {type_name}")

    return Dataset(data, class_name=class_name, name=name)
