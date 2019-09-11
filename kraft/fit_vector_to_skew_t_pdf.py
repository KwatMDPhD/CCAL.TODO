from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .ALMOST_ZERO import ALMOST_ZERO
from .check_array_for_bad import check_array_for_bad


def fit_vector_to_skew_t_pdf(vector, fit_initial_location=None, fit_initial_scale=None):

    vector_good = vector[~check_array_for_bad(vector, raise_for_bad=False)]

    fit_keyword_arguments = {}

    mean = vector_good.mean()

    if abs(mean) <= ALMOST_ZERO:

        mean = 0

    fit_keyword_arguments["loc"] = mean

    fit_keyword_arguments["scale"] = vector_good.std() / 2

    skew_t_model = ACSkewT_gen()

    degree_of_freedom, shape, location, scale = skew_t_model.fit(
        vector_good, **fit_keyword_arguments
    )

    if 24 < abs(shape):

        print("Refitting with fixed scale...")

        fit_keyword_arguments["fscale"] = fit_keyword_arguments["scale"]

        degree_of_freedom, shape, location, scale = skew_t_model.fit(
            vector_good, **fit_keyword_arguments
        )

        if 24 < abs(shape):

            print("Refitting with fixed location...")

            fit_keyword_arguments["floc"] = fit_keyword_arguments["loc"]

            degree_of_freedom, shape, location, scale = skew_t_model.fit(
                vector_good, **fit_keyword_arguments
            )

    return vector_good.size, location, scale, degree_of_freedom, shape
