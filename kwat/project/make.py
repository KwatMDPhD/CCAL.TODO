from os import mkdir
from os.path import join


def make(ro):

    mkdir(ro)

    for di in [
        "input/",
        "code/",
        "output/",
    ]:

        mkdir(join(ro, di))
