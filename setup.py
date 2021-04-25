from os import listdir

from setuptools import setup

n = "kraft"

d = "{}/data/".format(n)

setup(
    name=n,
    version="0.1.0",
    python_requires=">=3.6,<3.9",
    install_requires=(
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
    ),
    packages=(n,),
    package_data={n: tuple("{}{}".format(d, n) for n in listdir(path=d))},
)
