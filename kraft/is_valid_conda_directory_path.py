from os.path import isdir, join


def is_valid_conda_directory_path(conda_directory_path):

    return all(isdir(join(conda_directory_path, name)) for name in ("bin", "lib"))
