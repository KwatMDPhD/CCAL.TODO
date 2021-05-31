from numpy import apply_along_axis, asarray, concatenate, isnan, logical_not, median
from numpy.random import choice
from pandas import Series

from .grid import make_nd_grid
from .plot import plot_heat_map, plot_histogram


def summarize(
    da,
    pl=True,
    na="Name",
    he=int(1e6),
    hi=int(1e3),
):

    daar = da.to_numpy()

    la1_ = da.index.to_numpy()

    la2_ = da.columns.to_numpy()

    na1 = da.index.name

    na2 = da.columns.name

    print(daar.shape)

    title = {
        "text": na,
    }

    si = daar.size

    if pl and si <= he:

        plot_heat_map(
            da,
            layout={
                "title": title,
            },
        )

    bo_ = isnan(daar)

    n_na = bo_.sum()

    if 0 < n_na:

        print("% NaN: {:.2%}".format(n_na / si))

        if pl:

            plot_histogram(
                [
                    Series(bo_.sum(axis=1), la1_, name=na1),
                    Series(bo_.sum(axis=0), la2_, name=na2),
                ],
                layout={
                    "title": title,
                    "xaxis": {
                        "title": {
                            "text": "N NaN",
                        },
                    },
                },
            )

    if pl:

        plot_histogram(
            [
                Series(median(daar, axis=1), la1_, name=na1),
                Series(median(daar, axis=0), la2_, name=na2),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": {
                        "text": "(Not-NaN) Median",
                    },
                },
            },
        )

    bo_ = logical_not(bo_)

    daarno = daar[bo_]

    print("(Not-NaN) min: {:.2e}".format(daarno.min()))

    print("(Not-NaN) median: {:.2e}".format(median(daarno)))

    print("(Not-NaN) mean: {:.2e}".format(daarno.mean()))

    print("(Not-NaN) max: {:.2e}".format(daarno.max()))

    if pl:

        la_ = asarray(
            [
                "{}_{}".format(la1, la2)
                for la1, la2 in make_nd_grid([la1_, la2_])[bo_.ravel()]
            ]
        )

        if hi < daarno.size:

            print("Choosing {} for histogram...".format(hi))

            ie_ = concatenate(
                [
                    choice(daarno.size, hi, False),
                    [daarno.argmin(), daarno.argmax()],
                ]
            )

            daarno = daarno[ie_]

            la_ = la_[ie_]

        plot_histogram(
            [
                Series(daarno, la_, name="All"),
            ],
            layout={
                "title": title,
                "xaxis": {
                    "title": {
                        "text": "(Not-NaN) Number",
                    },
                },
            },
        )


def collapse(da):

    print(da.shape)

    print("Collapsing...")

    da = da.groupby(level=0).median()

    print(da.shape)

    return da


def process(
    feature_x_sample,
    drop_feature_=(),
    drop_sample_=(),
    nanize=None,
    drop_axis=None,
    drop_not_na_min_n=None,
    drop_not_na_unique_min_n=None,
    log_shift_minimum=None,
    log_base=None,
    normalize_axis=None,
    normalize_method=None,
    clip_min=None,
    clip_max=None,
    **kwarg_,
):

    if 0 < len(drop_feature_):

        print("Dropping {}: {}...".format(feature_x_sample.index.name, drop_feature_))

        feature_x_sample = feature_x_sample.drop(labels=drop_feature_, errors="ignore")

        summarize(feature_x_sample, **kwarg_)

    if 0 < len(drop_sample_):

        print("Dropping {}: {}...".format(feature_x_sample.columns.name, drop_sample_))

        feature_x_sample = feature_x_sample.drop(
            labels=drop_sample_, axis=1, errors="ignore"
        )

        summarize(feature_x_sample, **kwarg_)

    if nanize is not None:

        print("NaNizing <= {}...".format(nanize))

        matrix = feature_x_sample.to_numpy()

        matrix[matrix <= nanize] = nan

        feature_x_sample = DataFrame(
            data=matrix, index=feature_x_sample.index, columns=feature_x_sample.columns
        )

        summarize(feature_x_sample, **kwarg_)

    if drop_not_na_min_n is not None or drop_not_na_unique_min_n is not None:

        print("Dropping slice...")

        if drop_axis is None:

            drop_function = drop_axes_label

        else:

            drop_function = drop_axis_label

        shape = feature_x_sample.shape

        feature_x_sample = drop_function(
            feature_x_sample,
            drop_axis,
            not_na_min_n=drop_not_na_min_n,
            not_na_unique_min_n=drop_not_na_unique_min_n,
        )

        if shape != feature_x_sample.shape:

            summarize(feature_x_sample, **kwarg_)

    if log_base is not None:

        print(
            "Logging (log_min={}, log_base={})...".format(log_shift_minimum, log_base)
        )

        matrix = feature_x_sample.to_numpy()

        if log_shift_minimum is not None:

            matrix = shift_minimum(matrix, log_shift_minimum)

        feature_x_sample = DataFrame(
            data=log(matrix, base=log_base),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize(feature_x_sample, **kwarg_)

    if normalize_method is not None:

        print("Axis-{} {} normalizing...".format(normalize_axis, normalize_method))

        feature_x_sample = DataFrame(
            data=apply_along_axis(
                normalize, normalize_axis, feature_x_sample.to_numpy(), normalize_method
            ),
            index=feature_x_sample.index,
            columns=feature_x_sample.columns,
        )

        summarize(feature_x_sample, **kwarg_)

    if clip_min is not None or clip_max is not None:

        print("Clipping |{} - {}|...".format(clip_min, clip_max))

        feature_x_sample = feature_x_sample.clip(lower=clip_min, upper=clip_max)

        summarize(feature_x_sample, **kwarg_)

    return feature_x_sample


def separate_type(feature_x_sample, drop_constant=True, prefix_feature=True):

    continuous_row_ = []

    binary_x_sample_ = []

    for (_, row) in feature_x_sample.iterrows():

        try:

            is_continuous = (
                guess_type(row.dropna().astype(float).to_numpy()) == "continuous"
            )

        except ValueError:

            is_continuous = False

        if is_continuous:

            continuous_row_.append(row.apply(cast_builtin))

        elif not (drop_constant and row.unique().size == 1):

            binary_x_sample = binarize(row)

            if prefix_feature:

                label_template = "{}.{{}}".format(binary_x_sample.index.name)

            else:

                label_template = "{}"

            binary_x_sample.index = (
                label_template.format(label)
                for label in binary_x_sample.index.to_numpy()
            )

            binary_x_sample_.append(binary_x_sample)

    name_template = "{} ({{}})".format(feature_x_sample.index.name)

    if 0 < len(continuous_row_):

        continuous_x_sample = DataFrame(data=continuous_row_)

        continuous_x_sample.index.name = name_template.format("continuous")

    else:

        continuous_x_sample = None

    if 0 < len(binary_x_sample_):

        binary_x_sample = concat(binary_x_sample_)

        binary_x_sample.index.name = name_template.format("binary")

    else:

        binary_x_sample = None

    return (continuous_x_sample, binary_x_sample)
