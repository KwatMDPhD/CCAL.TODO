from kwat.path import get_absolute, select
from kwat.shell import shell_run


def run(pa):

    for na in select(get_absolute(pa), ke_=[r"^[^._].+\.ipynb$"]):

        shell_run(
            "jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --inplace {}".format(
                na
            )
        )
