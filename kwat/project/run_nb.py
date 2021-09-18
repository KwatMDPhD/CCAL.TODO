from kwat.path import get_absolute, select
from kwat.shell import run


def run_nb(pa):

    for na in select(get_absolute(pa), ke_=[r"^[^._].+\.ipynb$"]):

        run(
            "jupyter nbconvert --execute --ExecutePreprocessor.timeout=-1 --inplace {}".format(
                na
            )
        )
