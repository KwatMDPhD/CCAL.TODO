from click import Path, argument, group

from .workflow import make as workflow_make, run as workflow_run


@group()
def cli():
    pass


@cli.command()
@argument("name", required=True, nargs=1)
def make(name):
    """
    Make a workflow.
    """

    workflow_make(name)


@cli.command()
@argument("path", type=Path(exists=True), required=True, nargs=1)
def run(path):
    """
    Run all .ipynb in order.
    """

    workflow_run(path)
