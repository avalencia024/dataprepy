# dataprepy

Módulo de Python con herramientas básicas para el preprocesado de datasets,
desarrollado para la asignatura *Software Matemático y Estadístico*. Es la
versión en Python del paquete de R **dataprepR**, con las mismas funciones.
Todos los algoritmos están implementados a mano: solo se usan las
operaciones básicas de Python y numpy, pandas para gestionar las tablas de
datos y matplotlib para los gráficos.

## Instalación

Desde el fichero `.whl` (o el `.tar.gz`):

```
pip install dataprepy-1.0.0-py3-none-any.whl
```

Desde esta carpeta (la del código fuente):

```
pip install -e .
```

Necesita Python 3.9 o posterior. Las dependencias (numpy, pandas y
matplotlib) se instalan automáticamente si no están ya instaladas.

## Contenido

| Funcionalidad | Funciones |
|:--------------|:----------|
| Discretización | `discretize`, `discretizeEW`, `discretizeEF`, `discretizeData` |
| Métricas | `entropy`, `calculateAUC`, `computeMetrics` |
| Normalización y estandarización | `normalize`, `standardize`, `normalizeData`, `standardizeData` |
| Filtrado de variables | `filterData` |
| Correlación e información mutua | `correlation`, `mutualInformation`, `computeCorrelations` |
| Visualización | `plotROC`, `plotAUC`, `plotCorrelation`, `plotEntropy` |
| Gestión de datasets | `Dataset`, `isDataset`, `print`, `summary`, `readCSV`, `writeDataset`, `readDataset` |

## Ejemplo rápido

```python
import dataprepy as dp

coches = dp.readCSV("notebook/mtcars.csv")
ds = dp.Dataset(coches, class_name="am", name="Coches")
dp.computeMetrics(ds)                              # varianza, AUC y entropía
ds_filtrado = dp.filterData(ds, metric="auc", threshold=0.8)
dp.plotCorrelation(ds)                             # matriz de correlaciones
```

El notebook `notebook/dataprepy.ipynb` explica con detalle el uso de todas las
funciones y las decisiones de diseño tomadas. La documentación de cada
función se consulta con `help`, por ejemplo `help(dp.discretizeEF)`.

## Pruebas

Las pruebas usan pytest y están dentro del paquete (`dataprepy/test`). Con el
paquete ya instalado se ejecutan con:

```
pytest --pyargs dataprepy
```

## Estructura del proyecto

```
dataprepy/
├── setup.py         configuración para instalar el paquete (setuptools)
├── README.md        este fichero
├── LICENSE.txt      licencia (GPL 2 o posterior)
├── CHANGES.txt      cambios de cada versión
├── MANIFEST.in      ficheros adicionales que se incluyen en el .tar.gz
├── dataprepy/       código (organizado por funcionalidades)
│   └── test/        pruebas (pytest) y datos que usan
└── notebook/        notebook con ejemplos de uso y los datos que usa
```
