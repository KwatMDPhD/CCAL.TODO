from os import environ
from os.path import isdir, join
from urllib.parse import urljoin

from .internet import download
from .support import command, get_machine


def add_conda_to_path(conda_directory_path):

    bin_directory_path = join(conda_directory_path, "bin")

    environment_path = environ["PATH"]

    environ["PATH"] = "{}:{}".format(bin_directory_path, environment_path)


def get_conda_environments():

    environments = {}

    for line in command("conda-env list").stdout.strip().split(sep="\n"):

        if not line.startswith("#"):

            environment, path = (split for split in line.split() if split != "*")

            environments[environment] = {"path": path}

    return environments


def get_conda_prefix():

    return environ.get("CONDA_PREFIX")


def install_and_activate_conda(
    conda_directory_path, pip_installs=None, conda_installs=None
):

    if not is_conda_directory_path(conda_directory_path):

        conda_script_file_name_template = "Miniconda3-latest-{}-x86_64.sh"

        machine = get_machine()

        if "Darwin" in machine:

            str_ = "MacOSX"

        elif "Linux" in machine:

            str_ = "Linux"

        conda_script_file_name = conda_script_file_name_template.format(str_)

        tmp_directory_path = join("/", "tmp")

        download(
            urljoin("https://repo.continuum.io/miniconda/", conda_script_file_name),
            tmp_directory_path,
        )

        command(
            "bash {} -b -p {}".format(
                join(tmp_directory_path, conda_script_file_name), conda_directory_path
            )
        )

    add_conda_to_path(conda_directory_path)

    if pip_installs is not None:

        command("pip install {}".format(" ".join(pip_installs)))

    if conda_installs is not None:

        for channel, packages in conda_installs.items():

            command(
                "conda install --channel {} --yes {}".format(
                    channel, " ".join(packages)
                )
            )


def is_conda_directory_path(conda_directory_path):

    return all(
        isdir(join(conda_directory_path, directory_name))
        for directory_name in ("bin", "conda-meta", "lib")
    )
