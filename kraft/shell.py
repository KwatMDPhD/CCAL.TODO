from subprocess import PIPE, run as subprocess_run


def run(co):

    print(co)

    return subprocess_run(
        co, shell=True, stdout=PIPE, stderr=PIPE, check=True, universal_newlines=True
    )
