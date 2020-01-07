from numpy import full, nan
from pandas import DataFrame

from .fit_vector_to_skew_t_pdf import fit_vector_to_skew_t_pdf


def fit_each_dataframe_row_to_skew_t_pdf_(dataframe):

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
