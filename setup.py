from setuptools import find_packages, setup

name = "analyze"

setup(
    name=name,
    version="0.0.1",
    url="https://github.com/KwatME/{}".format(name),
    author="Kwat Medetgul-Ernar",
    author_email="kwatme8@gmail.com",
    packages=find_packages(),
    python_requires=">=3.6",
    package_data={name: ["data"]},
)
