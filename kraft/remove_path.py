from os import remove
from os.path import exists, isdir
from shutil import rmtree


def remove_path(path):

    if isdir(path):

        rmtree(path)

    elif exists(path):

        remove(path)

    print(f"Removed {path}.")
