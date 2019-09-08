from os.path import exists

from .run_command import run_command


def remove_path(path):

    if exists(path):

        run_command("rm -rf {}".format(path))

    else:

        print("{} does not exist for removal.".format(path))
