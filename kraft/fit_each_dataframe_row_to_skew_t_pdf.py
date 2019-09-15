from pandas import concat

from .call_function_with_multiprocess import call_function_with_multiprocess
from .fit_each_dataframe_row_to_skew_t_pdf_ import fit_each_dataframe_row_to_skew_t_pdf_
from .split_dataframe import split_dataframe


def fit_each_dataframe_row_to_skew_t_pdf(dataframe, n_job=1, output_tsv_file_path=None):

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

    if output_tsv_file_path is not None:

        skew_t_pdf_fit_parameter.to_csv(output_tsv_file_path, sep="\t")

    return skew_t_pdf_fit_parameter
