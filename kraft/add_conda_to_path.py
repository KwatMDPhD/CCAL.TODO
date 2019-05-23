from os import environ
from os.path import join


def add_conda_to_path(conda_directory_path):

    environ["PATH"] = f"{join(conda_directory_path, 'bin')}:{environ['PATH']}"
