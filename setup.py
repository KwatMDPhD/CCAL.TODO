from os import listdir

from setuptools import setup

from kraft.CONSTANT import DATA_DIRECTORY_PATH

name = "kraft"

setup(
    name=name,
    version="0.2.0",
    python_requires=">=3.6",
    install_requires=(
        "numpy",
        "pandas",
        "xlrd",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "KDEpy",
        "plotly",
        "click",
    ),
    packages=(name,),
    package_data={
        name: tuple(
            "{}/{}".format(DATA_DIRECTORY_PATH, file_name)
            for file_name in listdir(path=DATA_DIRECTORY_PATH)
        )
    },
)
