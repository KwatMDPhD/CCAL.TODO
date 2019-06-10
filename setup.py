from os.path import join

from setuptools import setup

NAME = "kraft"

VERSION = "0.0.2"

URL = f"https://github.com/KwatME/{NAME}"

setup(
    name=NAME,
    version=VERSION,
    url=URL,
    author="Kwat Medetgul-Ernar",
    author_email="kwatme8@gmail.com",
    python_requires=">=3.7",
    install_requires=(
        "numpy",
        "pandas",
        "scipy==1.2.1",
        "scikit-learn",
        "statsmodels",
        "KDEpy",
        "tables",
        "seaborn",
        "plotly",
        "GEOparse",
        "click",
        "pyyaml",
    ),
    packages=(NAME,),
    package_data={
        NAME: (
            join("data", "cell_line_name_best_cell_line_name.tsv"),
            join("data", "hgnc.tsv"),
        )
    },
)
