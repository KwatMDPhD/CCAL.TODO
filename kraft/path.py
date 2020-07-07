from os import mkdir, walk
from os.path import isdir, split
from re import sub


def path(path):

    directory_path = split(path)[0]

    directory_paths_missing = []

    while directory_path != "" and not isdir(directory_path):

        directory_paths_missing.append(directory_path)

        directory_path = split(directory_path)[0]

    for directory_path in reversed(directory_paths_missing):

        mkdir(directory_path)

        print("Made {}/.".format(directory_path))


def get_child_paths(parent_directory_path, relative=True):

    child_paths = []

    for directory_path, directory_names, file_names in walk(parent_directory_path):

        for directory_name in directory_names:

            child_paths.append("{}/{}/".format(directory_path, directory_name))

        for file_name in file_names:

            child_paths.append("{}/{}".format(directory_path, file_name))

    if relative:

        n = len(parent_directory_path) + 1

        return tuple(child_path[n:] for child_path in child_paths)

    else:

        return tuple(child_paths)


def clean(file_name):

    file_name_clean = sub(
        r"(?u)[^-\w.]", "_", file_name.strip().lower().replace(" ", "_")
    )

    print("{} ==> {}".format(file_name, file_name_clean))

    return file_name_clean
