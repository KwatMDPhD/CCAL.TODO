from gzip import open as gzip_open
from pickle import load


def read(pa):

    with gzip_open(pa) as io:

        return load(io)
