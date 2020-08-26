from os import listdir, mkdir, walk
from os.path import dirname, isdir
from re import sub


def path(path):

    directory_path = dirname(path)

    missing_directory_paths = []

    while directory_path != "" and not isdir(directory_path):

        missing_directory_paths.append(directory_path)

        directory_path = dirname(directory_path)

    for directory_path in missing_directory_paths[::-1]:

        mkdir(directory_path)

        print("{}/".format(directory_path))


def get_child_paths(directory_path, absolute=True):

    child_paths = []

    for directory_path_, directory_names, file_names in walk(directory_path):

        for name in directory_names:

            child_paths.append("{}/{}/".format(directory_path_, name))

        for name in file_names:

            child_paths.append("{}/{}".format(directory_path_, name))

    if absolute:

        return tuple(child_paths)

    else:

        n = len(directory_path) + 1

        return tuple(child_path[n:] for child_path in child_paths)


def clean(name):

    name_ = sub(r"(?u)[^-\w.]", "_", name.strip().lower())

    print("{} ==> {}".format(name, name_))

    return name_


def list_(directory_path):

    return tuple(
        "{}{}".format(directory_path, name) for name in listdir(path=directory_path)
    )
