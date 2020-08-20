from platform import uname
from subprocess import PIPE, CalledProcessError, run


def get_machine():

    uname_ = uname()

    return "{}_{}".format(uname_.system, uname_.machine)


def get_environment():

    environemnt = {}

    # TODO: try [:-1] instead of strip

    for line in command("env").stdout.splitlines():

        if line != "" and not line.strip().startswith(":"):

            key, value = line.split(sep="=", maxsplit=1)

            environemnt[key.strip()] = value.strip()

    return environemnt


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


def check_is_installed(program):

    try:

        # TODO: try [:-1] instead of strip
        return bool(command("type {}".format(program)).stdout.strip())

    except CalledProcessError:

        return False


def install_python_libraries(libraries):

    # TODO: try [:-1] instead of strip
    libraries_now = tuple(
        line.split(maxsplit=1)[0].lower()
        for line in command("pip list").stdout.strip().splitlines()[2:]
    )

    for library in libraries:

        library = library.lower()

        if library not in libraries_now:

            command("pip install {}".format(library))
