from gzip import open as gzip_open
from pickle import dump


def write(pa, gp):
    with gzip_open(pa, mode="wb") as io:
        dump(gp, io)
