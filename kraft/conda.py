from os import environ
from os.path import isdir

from .internet import download
from .shell import command, get_machine


def add_conda_to_path(directory_path):

    environ["PATH"] = "{}:{}".format("{}bin/".format(directory_path), environ["PATH"])


def get_conda_environments():

    environments = {}

    # TODO: try [:-1] instead of strip
    for line in command("conda-env list").stdout.strip().splitlines():

        if line[0] != "#":

            environment, path = (split for split in line.split() if split != "*")

            environments[environment] = {"path": path}

    return environments


def get_conda_prefix():

    return environ.get("CONDA_PREFIX")


def install_and_activate_conda(directory_path, pip_installs=None, conda_installs=None):

    if not is_conda(directory_path):

        machine = get_machine()

        if "Darwin" in machine:

            machine = "MacOSX"

        elif "Linux" in machine:

            machine = "Linux"

        command(
            "bash {} -b -p {}".format(
                download(
                    "https://repo.continuum.io/miniconda/Miniconda3-latest-{}-x86_64.sh".format(
                        machine
                    ),
                    "/tmp",
                ),
                directory_path,
            )
        )

    add_conda_to_path(directory_path)

    if pip_installs is not None:

        command("pip install {}".format(" ".join(pip_installs)))

    if conda_installs is not None:

        for channel, packages in conda_installs.items():

            command(
                "conda install --channel {} --yes {}".format(
                    channel, " ".join(packages)
                )
            )


def is_conda(directory_path):

    return all(
        isdir("{}{}".format(directory_path, name))
        for name in ("bin/", "conda-meta/", "lib/")
    )
