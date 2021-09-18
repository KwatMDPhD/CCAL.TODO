from click import Path, argument, group

from .project import make as workflow_make, run_nb as workflow_run_nb


@group()
def cli():
    pass


@cli.command()
@argument("path", type=Path(exists=True), required=True, nargs=1)
def run_nb(path):
    """
    Run all .ipynb in order.
    """

    workflow_run_nb(path)


@cli.command()
@argument("name", required=True, nargs=1)
def make(name):
    """
    Make a project.
    """

    workflow_make(name)
