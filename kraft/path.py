from os import mkdir, walk
from os.path import dirname, isdir
from re import sub


def path(path):

    directory_path = dirname(path)

    missing_directory_paths = []

    while directory_path != "" and not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        directory_path = dirname(directory_path)

    for directory_path in reversed(missing_directory_paths):

        mkdir(directory_path)

        print(directory_path)


def get_child_paths(parent_directory_path, absolute=True):

    child_paths = []

    for directory_path, directory_names, file_names in walk(parent_directory_path):

        for directory_name in directory_names:

            child_paths.append("{}/{}/".format(directory_path, directory_name))

        for file_name in file_names:

            child_paths.append("{}/{}".format(directory_path, file_name))

    if absolute:

        return tuple(child_paths)

    else:

        n = len(parent_directory_path) + 1

        return tuple(child_path[n:] for child_path in child_paths)


def clean(name):

    name_clean = sub(r"(?u)[^-\w.]", "_", name.strip().lower().replace(" ", "_"))

    print("{} ==> {}".format(name, name_clean))

    return name_clean
