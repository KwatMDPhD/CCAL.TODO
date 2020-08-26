from os.path import dirname

from numpy import finfo

RANDOM_SEED = 20121020

FLOAT_RESOLUTION = finfo(float).resolution

DATA_DIRECTORY_PATH = "{}/data/".format(dirname(__file__))

SAMPLE_FRACTION = 0.632

GOLDEN_FACTOR = 1.618
