from .is_str_version import is_str_version
from .run_command import run_command


def get_git_versions():

    tags = run_command("git tag --list").stdout.strip().split(sep="\n")

    versions = [tag for tag in tags if is_str_version(tag)]

    versions.sort(key=lambda iii: tuple(int(i) for i in iii.split(sep=".")))

    return versions
