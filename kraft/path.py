from os import mkdir
from os.path import isdir, split


def path(path, path_type):

    directory_path, file_name = split(path)

    missing_directory_paths = []

    while directory_path != "" and not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        directory_path, file_name = split(directory_path)

    for directory_path in reversed(missing_directory_paths):

        mkdir(directory_path)

        print("Made {}/.".format(directory_path))
