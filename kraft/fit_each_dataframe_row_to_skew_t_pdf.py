from numpy import full, nan
from pandas import DataFrame, concat

from .call_function_with_multiprocess import call_function_with_multiprocess
from .fit_vector_to_skew_t_pdf import fit_vector_to_skew_t_pdf
from .split_dataframe import split_dataframe


def _fit_each_dataframe_row_to_skew_t_pdf(dataframe):

    skew_t_pdf_fit_parameter = full((dataframe.shape[0], 5), nan)

    n = dataframe.shape[0]

    n_per_print = max(1, n // 10)

    for i, (index, series) in enumerate(dataframe.iterrows()):

        if i % n_per_print == 0:

            print("({}/{}) {}...".format(i + 1, n, index))

        skew_t_pdf_fit_parameter[i] = fit_vector_to_skew_t_pdf(series.values)

    return DataFrame(
        skew_t_pdf_fit_parameter,
        index=dataframe.index,
        columns=("N Data", "Location", "Scale", "Degree of Freedom", "Shape"),
    )


def fit_each_dataframe_row_to_skew_t_pdf(dataframe, n_job=1, output_file_path=None):

    skew_t_pdf_fit_parameter = concat(
        call_function_with_multiprocess(
            _fit_each_dataframe_row_to_skew_t_pdf,
            (
                (dataframe_,)
                for dataframe_ in split_dataframe(
                    dataframe, 0, min(dataframe.shape[0], n_job)
                )
            ),
            n_job,
        )
    )

    if output_file_path is not None:

        skew_t_pdf_fit_parameter.to_csv(output_file_path, sep="\t")

    return skew_t_pdf_fit_parameter
