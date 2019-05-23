from os.path import join
from urllib.parse import urljoin

from .add_conda_to_path import add_conda_to_path
from .download_url import download_url
from .get_machine import get_machine
from .is_conda_directory_path import is_conda_directory_path
from .run_command import run_command


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

        download_url(
            urljoin("https://repo.continuum.io/miniconda/", conda_script_file_name),
            tmp_directory_path,
        )

        run_command(
            f"bash {join(tmp_directory_path, conda_script_file_name)} -b -p {conda_directory_path}"
        )

    add_conda_to_path(conda_directory_path)

    if pip_installs is not None:

        run_command(f"pip install {' '.join(pip_installs)}")

    if conda_installs is not None:

        for channel, packages in conda_installs.items():

            run_command(f"conda install --channel {channel} --yes {' '.join(packages)}")
