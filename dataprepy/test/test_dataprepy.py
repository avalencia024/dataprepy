# =============================================================================
# test_dataprepy.py
# -----------------------------------------------------------------------------
# Pruebas del paquete. Con el paquete instalado se ejecutan con:
#     pytest --pyargs dataprepy
# Comprueban ejemplos con resultados calculados a mano y que el resto de
# funciones dan lo esperado. Algunas funciones de numpy y
# pandas (np.corrcoef, value_counts) solo se usan aquí, para comparar.
# =============================================================================

import os

import matplotlib
matplotlib.use("Agg")  # para que los gráficos no abran ninguna ventana

import numpy as np
import pandas as pd
import pytest

import dataprepy as dp

# Datos de ejemplo, en la carpeta datos que está junto a este fichero
DATA_FOLDER = os.path.join(os.path.dirname(__file__), "datos")
MTCARS = pd.read_csv(os.path.join(DATA_FOLDER, "mtcars.csv"))
IRIS = dp.readCSV(os.path.join(DATA_FOLDER, "iris.csv"))

# Vector de ejemplo para la discretización
X = [11.5, 10.2, 1.2, 0.5, 5.3, 20.5, 8.4]


def test_discretizeEW():
    # Igual anchura con 4 intervalos: puntos de corte 5.5, 10.5 y 15.5
    x_discretized, cut_points = dp.discretizeEW(X, num_bins=4)
    assert list(cut_points) == [5.5, 10.5, 15.5]
    assert list(x_discretized) == ["(10.5, 15.5]", "(5.5, 10.5]", "(-inf, 5.5]",
                                   "(-inf, 5.5]", "(-inf, 5.5]", "(15.5, inf)",
                                   "(5.5, 10.5]"]

    # Valores nuevos con los mismos puntos de corte (un valor igual a un punto
    # de corte va al intervalo de la izquierda)
    new_values = dp.discretize([-3, 5.5, 12, 100], cut_points)
    assert list(new_values) == ["(-inf, 5.5]", "(-inf, 5.5]", "(10.5, 15.5]",
                                "(15.5, inf)"]

    # Con números muy pequeños los nombres de los intervalos no se repiten
    x_discretized, cut_points = dp.discretizeEW([0, 0.0001, 0.0002, 0.0003, 0.0004], 4)
    assert len(x_discretized.categories) == 4

    # Variable constante: un único intervalo (con aviso)
    with pytest.warns(UserWarning):
        x_discretized, cut_points = dp.discretizeEW([3, 3, 3], num_bins=3)
    assert list(x_discretized.categories) == ["(-inf, inf)"]

    with pytest.raises(ValueError):
        dp.discretizeEW(X, num_bins=1)
    with pytest.raises(ValueError):
        dp.discretizeEW([1, 2, np.inf], num_bins=2)


def test_discretizeEF():
    # 7 valores en 4 intervalos: 2, 2, 1 y 2 valores
    x_discretized, cut_points = dp.discretizeEF(X, num_bins=4)
    assert np.allclose(cut_points, [3.25, 9.3, 10.85])
    assert list(pd.Series(x_discretized).value_counts(sort=False)) == [2, 2, 1, 2]

    # Con valores repetidos los cuatro 1 van juntos, los cortes se ponen entre
    # valores distintos (1.5 y 3.5) y no queda ningún intervalo vacío. Salen 3
    # intervalos en vez de 4 (con aviso). Es el mismo resultado que en R.
    with pytest.warns(UserWarning):
        x_discretized, cut_points = dp.discretizeEF([1, 1, 1, 1, 2, 3, 4, 5], 4)
    assert list(cut_points) == [1.5, 3.5]
    assert list(pd.Series(x_discretized).value_counts(sort=False)) == [4, 2, 2]

    # Variable constante: un único intervalo (con aviso)
    with pytest.warns(UserWarning):
        x_discretized, cut_points = dp.discretizeEF([3, 3, 3], num_bins=2)
    assert len(x_discretized.categories) == 1


def test_discretizeData():
    iris_disc, cut_points = dp.discretizeData(IRIS, num_bins=3, method="EF")
    assert list(cut_points.keys()) == ["Sepal.Length", "Sepal.Width",
                                       "Petal.Length", "Petal.Width"]
    assert iris_disc["Sepal.Length"].dtype == "category"
    assert iris_disc["Species"].equals(IRIS["Species"])

    # En un Dataset la variable clase no se discretiza
    ds = dp.Dataset(MTCARS, class_name="am")
    ds_disc, cut_points = dp.discretizeData(ds, num_bins=3)
    assert ds_disc.data["am"].equals(MTCARS["am"])
    assert ds_disc.data["mpg"].dtype == "category"

    # Los datos originales no cambian
    assert MTCARS["mpg"].dtype == float


def test_entropy():
    # Calculado a mano: -0.4 * log2(0.4) - 0.6 * log2(0.6) = 0.971
    x = pd.Categorical(["a", "a", "c", "c", "c"], categories=["a", "b", "c"])
    assert round(dp.entropy(x), 3) == 0.971

    # Una sola categoría: 0. Cuatro categorías igual de frecuentes: log2(4) = 2
    assert dp.entropy(pd.Categorical(["a", "a", "a"])) == 0
    assert dp.entropy(pd.Categorical(["a", "b", "c", "d"])) == 2

    # No se admiten valores perdidos
    with pytest.raises(ValueError):
        dp.entropy(pd.Categorical(["a", None]))


def test_calculateAUC():
    data = pd.DataFrame({"value": [0.1, 0.4, 0.35, 0.8],
                         "class": [False, False, True, True]})
    roc_points, auc = dp.calculateAUC(data)
    assert auc == 0.75
    assert list(roc_points["FPR"]) == [0, 0, 0.5, 0.5, 1]
    assert list(roc_points["TPR"]) == [0, 0.5, 0.5, 1, 1]

    # Separación perfecta (AUC = 1) e inversa (AUC = 0)
    labels = [False, False, True, True]
    assert dp.calculateAUC(pd.DataFrame({"v": [1, 2, 3, 4], "c": labels}))[1] == 1
    assert dp.calculateAUC(pd.DataFrame({"v": [4, 3, 2, 1], "c": labels}))[1] == 0

    # La clase también puede ser 0/1 (el positivo es el 1)
    assert dp.calculateAUC(MTCARS[["wt", "am"]])[1] == \
        dp.calculateAUC(pd.DataFrame({"wt": MTCARS["wt"], "am": MTCARS["am"] == 1}))[1]

    # En una clase categórica el positivo es la segunda categoría (como en R),
    # aunque por orden alfabético vaya antes
    clase = pd.Categorical(["positivo", "positivo", "negativo", "negativo"],
                           categories=["positivo", "negativo"])
    assert dp.calculateAUC(pd.DataFrame({"v": [1, 2, 3, 4], "c": clase}))[1] == 1

    # La clase no puede tener valores infinitos (ni perdidos)
    with pytest.raises(ValueError):
        dp.calculateAUC(pd.DataFrame({"v": [1, 2, 3, 4], "c": [0, np.inf, 0, np.inf]}))
    with pytest.raises(ValueError):
        dp.computeMetrics(pd.DataFrame({"v": [1, 2, 3, 4], "c": [0, np.inf, 0, np.inf]}),
                          class_name="c")


def test_computeMetrics():
    data = MTCARS.copy()
    data["cyl"] = data["cyl"].astype("category")
    metrics = dp.computeMetrics(data, class_name="am")

    assert metrics.loc["mpg", "type"] == "numeric"
    assert metrics.loc["cyl", "type"] == "categorical"
    assert np.isclose(metrics.loc["mpg", "variance"], np.var(MTCARS["mpg"], ddof=1))
    assert np.isclose(metrics.loc["drat", "auc"], 0.9028, atol=1e-4)
    assert np.isnan(metrics.loc["cyl", "auc"])
    assert "am" not in metrics.index

    # Clase con tres valores: se avisa y el AUC queda NaN
    with pytest.warns(UserWarning):
        metrics = dp.computeMetrics(IRIS, class_name="Species")
    assert metrics["auc"].isna().all()

    # No se admiten valores perdidos
    with pytest.raises(ValueError):
        dp.computeMetrics(pd.DataFrame({"a": [1.0, np.nan, 3.0]}))


def test_normalize_standardize():
    v = [-0.29, 2.11, -3.84, 10.69, 5.30]
    assert dp.normalize(v).min() == 0 and dp.normalize(v).max() == 1
    assert np.isclose(dp.standardize(v).mean(), 0)
    assert np.isclose(dp.standardize(v).std(ddof=1), 1)

    # Con un solo valor no se puede dividir: vector de ceros (con aviso)
    with pytest.warns(UserWarning):
        assert list(dp.standardize([5])) == [0]

    iris_norm = dp.normalizeData(IRIS)
    assert iris_norm["Petal.Length"].min() == 0
    assert iris_norm["Petal.Length"].max() == 1
    assert iris_norm["Species"].equals(IRIS["Species"])

    # En un Dataset la clase no se transforma
    ds = dp.Dataset(MTCARS, class_name="am")
    assert dp.standardizeData(ds).data["am"].equals(MTCARS["am"])


def test_filterData():
    data = MTCARS.copy()
    data["cyl"] = data["cyl"].astype("category")

    # Varianza mayor que 10: solo quedan mpg, disp y hp
    filtered = dp.filterData(data, metric="variance", threshold=10)
    assert list(filtered.columns) == ["mpg", "disp", "hp"]

    # Con keep_others=True se mantiene también la categórica cyl
    filtered = dp.filterData(data, metric="variance", threshold=10,
                             keep_others=True)
    assert list(filtered.columns) == ["mpg", "cyl", "disp", "hp"]

    # AUC mayor que 0.8 (aquí gear es numérica y también cumple el requisito).
    # La clase se mantiene siempre.
    filtered = dp.filterData(data, metric="auc", threshold=0.8, class_name="am")
    assert list(filtered.columns) == ["mpg", "drat", "am", "gear"]

    # Si ninguna variable cumple el requisito solo queda la clase, y sus
    # métricas son una tabla sin filas
    only_class = dp.filterData(dp.Dataset(data, class_name="am"),
                               metric="variance", threshold=1e6)
    assert list(only_class.data.columns) == ["am"]
    assert len(dp.computeMetrics(only_class)) == 0

    # Para filtrar por AUC hace falta la clase
    with pytest.raises(ValueError):
        dp.filterData(data, metric="auc", threshold=0.8)

    # El umbral tiene que ser un número finito
    with pytest.raises(ValueError):
        dp.filterData(data, metric="variance", threshold=np.nan)


def test_correlation_mutualInformation():
    assert np.isclose(dp.correlation(MTCARS["mpg"], MTCARS["wt"]),
                      np.corrcoef(MTCARS["mpg"], MTCARS["wt"])[0, 1])

    # Información mutua: 0 si son independientes y la entropía consigo misma
    a = pd.Categorical(["x", "x", "y", "y"])
    b = pd.Categorical(["u", "v", "u", "v"])
    assert dp.mutualInformation(a, b) == 0
    assert dp.mutualInformation(a, a) == dp.entropy(a)

    data = MTCARS.copy()
    data["cyl"] = data["cyl"].astype("category")
    data["gear"] = data["gear"].astype("category")
    correlations, mutual_info = dp.computeCorrelations(data)
    assert np.allclose(correlations.loc["mpg", "wt"],
                       np.corrcoef(MTCARS["mpg"], MTCARS["wt"])[0, 1])
    assert list(mutual_info.columns) == ["cyl", "gear"]


def test_plots():
    # Solo comprobamos que se dibujan sin errores y lo que devuelven
    data = pd.DataFrame({"value": [0.1, 0.4, 0.35, 0.8],
                         "class": [False, False, True, True]})
    assert dp.plotROC(data)[1] == 0.75
    assert len(dp.plotAUC(MTCARS, class_name="am")) == 10
    assert dp.plotCorrelation(MTCARS).shape == (11, 11)
    iris_disc, cut_points = dp.discretizeData(IRIS, num_bins=4)
    assert (dp.plotEntropy(iris_disc) <= 1).all()


def test_read_write(tmp_path):
    coches = dp.readCSV(os.path.join(DATA_FOLDER, "mtcars.csv"))
    assert coches["cyl"].dtype == "category"
    assert coches["hp"].dtype == "int64"
    assert coches["mpg"].dtype == float

    # Al guardar y volver a leer se recupera exactamente el mismo dataset
    file_path = os.path.join(tmp_path, "coches.dts")
    ds = dp.Dataset(coches, class_name="am", name="Coches")
    dp.writeDataset(ds, file_path)
    ds2 = dp.readDataset(file_path)
    assert ds2.name == "Coches" and ds2.class_name == "am"
    assert ds2.data.equals(ds.data)

    # También con textos con comas, tildes o que parecen valores perdidos
    # ("NA", "N/A", "") y una variable lógica
    data = pd.DataFrame({"texto": ["hola", "con, coma", "eñe", "NA", "N/A", ""],
                         "logica": [True, False, True, True, False, True]})
    dp.writeDataset(data, file_path)
    assert dp.readDataset(file_path).data.equals(data)

    # El formato no admite valores perdidos
    with pytest.raises(ValueError):
        dp.writeDataset(pd.DataFrame({"x": [1.0, np.nan]}), file_path)


def test_Dataset():
    ds = dp.Dataset(MTCARS, class_name="am", name="Coches")
    assert "32 filas x 11 variables" in str(ds)
    assert dp.isDataset(ds)
    assert not dp.isDataset(MTCARS)
    assert ds.summary().equals(dp.computeMetrics(MTCARS, class_name="am"))

    with pytest.raises(ValueError):
        dp.Dataset(MTCARS, class_name="no_existe")
