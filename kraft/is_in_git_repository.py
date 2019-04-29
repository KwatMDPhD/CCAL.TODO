from subprocess import CalledProcessError

from .run_command import run_command


def is_in_git_repository():

    try:

        run_command("git status")

        return True

    except CalledProcessError:

        return False
