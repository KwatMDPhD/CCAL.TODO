from gzip import open as gzip_open
from pickle import load


def read_gps_map(pickle_gz_file_path):

    with gzip_open(pickle_gz_file_path) as io:

        return load(io)
