from os import environ
from os.path import join


def add_conda_to_path(conda_directory_path):

    bin_directory_path = join(conda_directory_path, "bin")

    environment_path = environ["PATH"]

    environ["PATH"] = "{}:{}".format(bin_directory_path, environment_path)
