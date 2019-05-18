from os.path import exists

from .run_command import run_command


def remove_path(path):

    if exists(path):

        run_command("rm --recursive --force path")

    else:

        print(f"{path} does not exist for removal.")
