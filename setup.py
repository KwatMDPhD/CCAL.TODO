from os import listdir

from setuptools import setup

name = "kraft"

data_directory_path = "{}/data/".format(name)

setup(
    name=name,
    version="0.1.0",
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
        "requests",
    ),
    packages=(name,),
    package_data={
        name: tuple(
            "{}/{}".format(data_directory_path, file_name)
            for file_name in listdir(path=data_directory_path)
        )
    },
)
