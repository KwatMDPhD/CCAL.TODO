from os.path import dirname

from numpy import finfo

from . import (
    array,
    clustering,
    dataframe,
    dict_,
    feature_x_sample,
    function_heat_map,
    gct,
    geo,
    gmt,
    grid,
    information,
    internet,
    iterable,
    json,
    kernel_density,
    name_biology,
    path,
    plot,
    probability,
    sea,
    series,
    shell,
    significance,
    str_,
    support,
)

RANDOM_SEED = 20121020

FLOAT_RESOLUTION = finfo(float).resolution

DATA_DIRECTORY_PATH = "{}/data/".format(dirname(__file__))
