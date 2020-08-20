from numpy import full, nan
from pandas import DataFrame, concat
from statsmodels.sandbox.distributions.extras import ACSkewT_gen

from .CONSTANT import FLOAT_RESOLUTION


def fit_each_dataframe_row_to_skew_t_pdf(dataframe, n_job=1, tsv_file_path=None):

    skew_t_pdf_fit_parameter = concat(
        call_function_with_multiprocess(
            fit_each_dataframe_row_to_skew_t_pdf_,
            (
                (dataframe_,)
                for dataframe_ in split_dataframe(
                    dataframe, 0, min(dataframe.shape[0], n_job)
                )
            ),
            n_job,
        )
    )

    if tsv_file_path is not None:

        skew_t_pdf_fit_parameter.to_csv(tsv_file_path, sep="\t")

    return skew_t_pdf_fit_parameter


def fit_each_dataframe_row_to_skew_t_pdf_(dataframe):

    skew_t_pdf_fit_parameter = full((dataframe.shape[0], 5), nan)

    n = dataframe.shape[0]

    for i, (index, series) in enumerate(dataframe.iterrows()):

        skew_t_pdf_fit_parameter[i] = fit_vector_to_skew_t_pdf(series.to_numpy())

    return DataFrame(
        data=skew_t_pdf_fit_parameter,
        index=dataframe.index,
        columns=("N Data", "Location", "Scale", "Degree of Freedom", "Shape"),
    )


def fit_vector_to_skew_t_pdf(vector, fit_initial_location=None, fit_initial_scale=None):

    vector_good = vector[~check_array_for_bad(vector, raise_for_bad=False)]

    fit_keyword_arguments = {}

    mean = vector_good.mean()

    if abs(mean) <= FLOAT_RESOLUTION:

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
