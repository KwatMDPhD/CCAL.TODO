from os.path import isdir, join


def is_conda_directory_path(conda_directory_path):

    return all(
        isdir(join(conda_directory_path, directory_name))
        for directory_name in ("bin", "conda-meta", "lib")
    )
