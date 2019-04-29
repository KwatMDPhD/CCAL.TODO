from multiprocessing.pool import Pool

from numpy.random import seed

from .RANDOM_SEED import RANDOM_SEED


def call_function_with_multiprocess(
    function, arguments, n_job, random_seed=RANDOM_SEED
):

    seed(seed=random_seed)

    with Pool(n_job) as process:

        return process.starmap(function, arguments)
