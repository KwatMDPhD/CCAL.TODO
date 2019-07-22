from os.path import join

from setuptools import setup

from kraft import get_child_paths

NAME = "kraft"

VERSION = "0.0.5"

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
        "xlrd",
        "scipy==1.2.1",
        "scikit-learn",
        "statsmodels",
        "KDEpy",
        "tables",
        "seaborn",
        "plotly",
        "chart_studio",
        "GEOparse",
        "click",
    ),
    packages=(NAME,),
    package_data={
        NAME: tuple(join("data", path) for path in get_child_paths(join(NAME, "data")))
    },
)
