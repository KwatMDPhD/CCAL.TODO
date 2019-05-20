from subprocess import CalledProcessError

from .run_command import run_command


def is_program(program_name):

    try:

        return bool(run_command(f"type {program_name}").stdout.strip())

    except CalledProcessError:

        return False
