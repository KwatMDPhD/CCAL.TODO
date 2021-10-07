from os.path import join
from shutil import copytree

from ..constant import DATA_DIRECTORY_PATH


def make(na):

    copytree(join(DATA_DIRECTORY_PATH, "workflow", ""), na)
