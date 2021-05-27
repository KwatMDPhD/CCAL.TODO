from subprocess import PIPE, run


def command(c):

    print(c)

    return run(
        c,
        shell=True,
        stdout=PIPE,
        stderr=PIPE,
        check=True,
        universal_newlines=True,
    )
