from os import walk
from os.path import join

from setuptools import setup

name = "kraft"


def get_child_paths(parent_directory_path, relative=True):

    child_paths = []

    for directory_path, directory_names, file_names in walk(parent_directory_path):

        for directory_name in directory_names:

            child_paths.append(join(directory_path, "{}/".format(directory_name)))

        for file_name in file_names:

            child_paths.append(join(directory_path, file_name))

    if relative:

        n = len(parent_directory_path) + 1

        return tuple(child_path[n:] for child_path in child_paths)

    else:

        return tuple(child_paths)


setup(
    name=name,
    url="https://github.com/KwatME/{}".format(name),
    version="0.2.0",
    author="Kwat Medetgul-Ernar",
    author_email="kwatme8@gmail.com",
    python_requires=">=3.7",
    install_requires=(
        "numpy",
        "pandas",
        "xlrd",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "KDEpy",
        "tables",
        "plotly",
        "GEOparse",
        "click",
    ),
    packages=(name,),
    package_data={
        name: tuple(join("data", path) for path in get_child_paths(join(name, "data")))
    },
)
