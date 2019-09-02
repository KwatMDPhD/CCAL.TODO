from numpy import full, nan
from pandas import DataFrame, concat

from .call_function_with_multiprocess import call_function_with_multiprocess
from .fit_skew_t_pdf_on_1d_array import fit_skew_t_pdf_on_1d_array
from .split_dataframe import split_dataframe


def _fit_skew_t_pdf_on_each_dataframe_row(dataframe):

    skew_t_pdf_fit_parameter = full((dataframe.shape[0], 5), nan)

    n = dataframe.shape[0]

    n_per_print = max(1, n // 10)

    for i, (index, series) in enumerate(dataframe.iterrows()):

        if i % n_per_print == 0:

            print(f"({i + 1}/{n}) {index} ...")

        _1d_array = series.values

        skew_t_pdf_fit_parameter[i] = fit_skew_t_pdf_on_1d_array(_1d_array)

    return DataFrame(
        skew_t_pdf_fit_parameter,
        index=dataframe.index,
        columns=("N Data", "Location", "Scale", "Degree of Freedom", "Shape"),
    )


def fit_skew_t_pdf_on_each_dataframe_row(dataframe, n_job=1, output_file_path=None):

    skew_t_pdf_fit_parameter = concat(
        call_function_with_multiprocess(
            _fit_skew_t_pdf_on_each_dataframe_row,
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
