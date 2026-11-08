# setup.py: configuración para instalar el paquete con setuptools (como el
# fichero DESCRIPTION de los paquetes de R). Desde esta carpeta:
#     pip install -e .

from setuptools import setup

setup(
    name="dataprepy",
    version="1.0.0",
    author="Alazne Valencia Saez",
    author_email="avalencia024@ikasle.ehu.eus",
    packages=["dataprepy", "dataprepy.test"],
    package_data={"dataprepy.test": ["datos/*.csv"]},
    license="GPL-2.0-or-later",
    description="Herramientas básicas para el preprocesado de datasets",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.9",
    install_requires=[
        "numpy >= 1.23",
        "pandas >= 1.5",
        "matplotlib >= 3.5",
    ],
)
