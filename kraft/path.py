from os import listdir, mkdir, walk
from os.path import dirname, isdir
from re import sub


def make(path):

    directory_path = dirname(path)

    missing_directory_path_ = []

    while directory_path != "" and not isdir(directory_path):

        missing_directory_path_.append(directory_path)

        directory_path = dirname(directory_path)

    for directory_path in missing_directory_path_[::-1]:

        mkdir(directory_path)

        print("{}/".format(directory_path))


def get_child_path(directory_path, absolute=True):

    child_path_ = []

    for _directory_path, directory_name_, file_name_ in walk(directory_path):

        for name in directory_name_:

            child_path_.append("{}/{}/".format(_directory_path, name))

        for name in file_name_:

            child_path_.append("{}/{}".format(_directory_path, name))

    if absolute:

        return tuple(child_path_)

    else:

        n = len(directory_path) + 1

        return tuple(child_path[n:] for child_path in child_path_)


def clean(name):

    name_clean = sub(r"(?u)[^-\w.]", "_", name.strip().lower())

    print("{} => {}".format(name, name_clean))

    return name_clean


def list(directory_path):

    return tuple(
        "{}{}".format(directory_path, name) for name in listdir(make=directory_path)
    )
