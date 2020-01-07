from numpy import nan
from pandas import DataFrame

from .drop_dataframe_slice import drop_dataframe_slice
from .drop_dataframe_slice_greedily import drop_dataframe_slice_greedily
from .log_array import log_array
from .normalize_dataframe import normalize_dataframe
from .summarize_dataframe import summarize_dataframe


def process_dataframe(
    dataframe,
    axis0_elements_to_drop=(),
    axis1_elements_to_drop=(),
    naclip_min=None,
    drop_axis=None,
    max_na=None,
    min_n_not_na_value=None,
    min_n_not_na_unique_value=None,
    shift_as_necessary_to_achieve_min_before_logging=None,
    log_base=None,
    normalization_axis=None,
    normalization_method=None,
    clip_min=None,
    clip_max=None,
    **summarize_dataframe_keyword_arguments,
):

    assert not dataframe.index.has_duplicates

    assert not dataframe.columns.has_duplicates

    summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if 0 < len(axis0_elements_to_drop):

        print("Dropping {}: {}...".format(dataframe.index.name, axis0_elements_to_drop))

        dataframe.drop(axis0_elements_to_drop, errors="ignore", inplace=True)

        summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if 0 < len(axis1_elements_to_drop):

        print(
            "Dropping {}: {}...".format(dataframe.columns.name, axis1_elements_to_drop)
        )

        dataframe.drop(axis1_elements_to_drop, axis=1, errors="ignore", inplace=True)

        summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if naclip_min is not None:

        print("NA-clipping {}|...".format(naclip_min))

        dataframe[dataframe <= naclip_min] = nan

        summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if (
        max_na is not None
        or min_n_not_na_value is not None
        or min_n_not_na_unique_value is not None
    ):

        print("Dropping axis-{} slice...".format(drop_axis))

        if drop_axis is None:

            drop_function = drop_dataframe_slice_greedily

        else:

            drop_function = drop_dataframe_slice

        dataframe_shape_before_drop = dataframe.shape

        dataframe = drop_function(
            dataframe,
            drop_axis,
            max_na=max_na,
            min_n_not_na_value=min_n_not_na_value,
            min_n_not_na_unique_value=min_n_not_na_unique_value,
        )

        if dataframe.shape != dataframe_shape_before_drop:

            summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if log_base is not None:

        print(
            "Logging (shift_as_necessary_to_achieve_min_before_logging={}, log_base={})...".format(
                shift_as_necessary_to_achieve_min_before_logging, log_base
            )
        )

        dataframe = DataFrame(
            log_array(
                dataframe.values,
                raise_if_bad=False,
                shift_as_necessary_to_achieve_min_before_logging=shift_as_necessary_to_achieve_min_before_logging,
                log_base=log_base,
            ),
            index=dataframe.index,
            columns=dataframe.columns,
        )

        summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if normalization_method is not None:

        print(
            "Axis-{} {} normalizing...".format(normalization_axis, normalization_method)
        )

        dataframe = normalize_dataframe(
            dataframe, normalization_axis, normalization_method
        )

        summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    if clip_min is not None or clip_max is not None:

        print("Clipping |{} - {}|...".format(clip_min, clip_max))

        dataframe.clip(lower=clip_min, upper=clip_max, inplace=True)

        summarize_dataframe(dataframe, **summarize_dataframe_keyword_arguments)

    return dataframe
