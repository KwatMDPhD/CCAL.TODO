from subprocess import PIPE, run


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
