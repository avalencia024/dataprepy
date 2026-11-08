"""
dataprepy: herramientas básicas para el preprocesado de datasets.

Paquete desarrollado para la asignatura Software Matemático y Estadístico. Es
la versión en Python del paquete de R dataprepR, con las mismas funciones:

- Discretización por igual anchura e igual frecuencia: discretizeEW,
  discretizeEF, discretize y discretizeData.
- Métricas (varianza, AUC y entropía): computeMetrics, calculateAUC y entropy.
- Normalización y estandarización: normalize, standardize, normalizeData y
  standardizeData.
- Filtrado de variables según sus métricas: filterData.
- Correlación e información mutua: computeCorrelations, correlation y
  mutualInformation.
- Visualización: plotROC, plotAUC, plotCorrelation y plotEntropy.
- Gestión de datasets: clase Dataset (Dataset, isDataset) y lectura y
  escritura de datos (readCSV, writeDataset y readDataset).

La ayuda de cada función se consulta con help, por ejemplo
help(dataprepy.discretizeEF).

Autora: Alazne Valencia Saez
"""

from .correlation import computeCorrelations, correlation, mutualInformation
from .dataset import Dataset, isDataset
from .discretization import discretize, discretizeData, discretizeEF, discretizeEW
from .filtering import filterData
from .metrics import calculateAUC, computeMetrics, entropy
from .normalization import normalize, normalizeData, standardize, standardizeData
from .plotting import plotAUC, plotCorrelation, plotEntropy, plotROC
from .readwrite import readCSV, readDataset, writeDataset

__version__ = "1.0.0"
