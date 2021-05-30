from os import listdir

from setuptools import setup

na = "kraft"

pa = "{}/data/".format(na)

setup(
    name=na,
    version="0.1.0",
    python_requires=">=3.6,<3.9",
    install_requires=[
        "numpy",
        "pandas",
        "xlrd",
        "openpyxl",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "KDEpy",
        "plotly",
        "click",
        "requests",
    ],
    packages=[na],
    package_data={na: ["{}{}".format(pa, na) for na in listdir(pa) if na[0] != "."]},
)
