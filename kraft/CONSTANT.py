from os.path import dirname

from numpy import finfo

RANDOM_SEED = 20121020

FLOAT_RESOLUTION = finfo(float).resolution


DATA_DIRECTORY_PATH = "{}/../data".format(dirname(__file__))
