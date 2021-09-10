from click import Path, argument, group

from .workflow import run_nb as workflow_run_nb


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
