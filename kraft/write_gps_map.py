from gzip import open as gzip_open
from pickle import dump


def write_gps_map(gps_map, pickle_gz_file_path):

    with gzip_open(pickle_gz_file_path, mode="wb") as pickle_gz_file:

        dump(gps_map, pickle_gz_file)
