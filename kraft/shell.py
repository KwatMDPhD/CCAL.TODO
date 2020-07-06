from platform import uname
from subprocess import PIPE, CalledProcessError, run


def command(command):

    print(command)

    return run(
        command,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        check=True,
        universal_newlines=True,
    )


def get_machine():

    uname_ = uname()

    return "{}_{}".format(uname_.system, uname_.machine)


def get_shell_environment():

    environemnt = {}

    for line in command("env").stdout.split(sep="\n"):

        if line != "" and not line.strip().startswith(":"):

            key, value = line.split(sep="=", maxsplit=1)

            environemnt[key.strip()] = value.strip()

    return environemnt


def check_is_installed(program):

    try:

        return bool(command("type {}".format(program)).stdout.strip())

    except CalledProcessError:

        return False


def install_python_libraries(libraries):

    libraries_installed = tuple(
        line.split()[0]
        for line in command("pip list").stdout.strip().split(sep="\n")[2:]
    )

    for library in libraries:

        if library not in libraries_installed:

            command("pip install {}".format(library))
