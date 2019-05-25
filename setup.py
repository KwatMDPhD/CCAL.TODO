from distutils.core import setup
from os.path import join

from kraft import get_child_paths

NAME = "kraft"

VERSION = "0.0.2"

URL = "https://github.com/KwatME/kraft"

setup(
    name=NAME,
    version=VERSION,
    url=URL,
    author="Kwat Medetgul-Ernar",
    author_email="kwatme8@gmail.com",
    license="LICENSE",
    python_requires=">=3.6",
    install_requires=(),
    packages=(NAME,),
    package_data={
        NAME: tuple(join("data", path) for path in get_child_paths(join(NAME, "data")))
    },
)
