from os.path import dirname

from numpy import finfo

DATA_DIRECTORY_PATH = "{}/data/".format(dirname(__file__))

FLOAT_RESOLUTION = finfo(float).resolution

GOLDEN_FACTOR = 1.618

RANDOM_SEED = 20121020

SAMPLE_FRACTION = 0.632
