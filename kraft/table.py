from numpy import apply_along_axis, asarray, full, isnan, nan, unique
from numpy.random import choice, seed
from pandas import DataFrame, Index, isna, notna

from .array import map_integer
from .CONSTANT import RANDOM_SEED


def error(ta):

    for la_ in ta.axes:

        assert not la_.isna().any(), "Has Na."

        assert not la_.has_duplicates, "Duplicated."


def sync(df_, axis):

    df_0 = df_[0]

    label_ = df_0.axes[axis]

    for df in df_[1:]:

        label_ = label_.intersection(df.axes[axis])

    label_ = asarray(sorted(label_))

    return tuple(df.reindex(labels=label_, axis=axis) for df in df_)


def count(ta):

    for la, an_ in ta.iteritems():

        print("-" * 80)

        print(la)

        an_co = an_.value_counts()

        print(an_co)

        print("-" * 80)


def _check_has_enough_not_na(un_, n_no):

    return n_no <= notna(un_).sum()


def _check_has_enough_not_na_unique(nu_, n_un):

    return n_un <= unique(nu_[notna(nu_)]).si


def sample(df, axis_0_label_n, axis_1_label_n, random_seed=RANDOM_SEED, **kwarg_):

    (axis_0_size, axis_1_size) = df.shape

    seed(seed=random_seed)

    if axis_0_label_n is not None:

        if axis_0_label_n < 1:

            axis_0_label_n = int(axis_0_label_n * axis_0_size)

        axis_0_index_ = choice(axis_0_size, size=axis_0_label_n, **kwarg_)

    if axis_1_label_n is not None:

        if axis_1_label_n < 1:

            axis_1_label_n = int(axis_1_label_n * axis_1_size)

        axis_1_index_ = choice(axis_1_size, size=axis_1_label_n, **kwarg_)

    matrix = df.to_numpy()

    axis_0_label_ = df.index.to_numpy()

    axis_1_label_ = df.columns.to_numpy()

    axis_0_name = df.index.name

    axis_1_name = df.columns.name

    if axis_0_label_n is not None and axis_1_label_n is not None:

        return DataFrame(
            data=matrix[axis_0_index_, axis_1_index_],
            index=Index(data=axis_0_label_[axis_0_index_], name=axis_0_name),
            columns=Index(data=axis_1_label_[axis_1_index_], name=axis_1_name),
        )

    elif axis_0_label_n is not None:

        return DataFrame(
            data=matrix[axis_0_index_],
            index=Index(data=axis_0_label_[axis_0_index_], name=axis_0_name),
            columns=Index(data=axis_1_label_, name=axis_1_name),
        )

    elif axis_1_label_n is not None:

        return DataFrame(
            data=matrix[:, axis_1_index_],
            index=Index(data=axis_0_label_, name=axis_0_name),
            columns=Index(data=axis_1_label_[axis_1_index_], name=axis_1_name),
        )


def drop(da, ax, n_no=None, n_un=None):

    assert not (n_no is None and n_un is None)

    sh = da.shape

    bo_ = full(sh[ax], True)

    if ax == 0:

        axap = 1

    elif ax == 1:

        axap = 0

    daar = da.to_numpy()

    if n_no is not None:

        if n_no < 1:

            n_no *= sh[axap]

        bo_ &= apply_along_axis(_check_has_enough_not_na, axap, daar, n_no)

    if n_un is not None:

        if n_un < 1:

            n_un *= da.shape[axap]

        bo_ &= apply_along_axis(_check_has_enough_not_na_unique, axap, daar, n_un)

    if ax == 0:

        da = da.loc[bo_, :]

    elif ax == 1:

        da = da.loc[:, bo_]

    print("{} => {}".format(sh, da.shape))

    return da


def drop_both(da, ax=None, **ke):

    sh = da.shape

    if ax is None:

        ax = int(sh[0] < sh[1])

    re = False

    while True:

        da = drop(da, ax, **ke)

        sh2 = da.shape

        if re and sh == sh2:

            return da

        sh = sh2

        if ax == 0:

            ax = 1

        elif ax == 1:

            ax = 0

        re = True


def map_to(ta, co, fu=None):

    ke_va = {}

    for va, ke_ in zip(ta.pop(co).to_numpy(), ta.to_numpy()):

        for ke in ke_:

            if fu is None:

                ke_va[ke] = va

            else:

                for ke in fu(ke):

                    ke_va[ke] = va

    return ke_va


def pivot(
    axis_0_label_, axis_1_label_, value_, axis_0_name, axis_1_name, function=None
):

    axis_0_label_to_index = map_integer(axis_0_label_)[0]

    axis_1_label_to_index = map_integer(axis_1_label_)[0]

    matrix = full((len(axis_0_label_to_index), len(axis_1_label_to_index)), nan)

    for (axis_0_label, axis_1_label, value) in zip(
        axis_0_label_, axis_1_label_, value_
    ):

        axis_0_index = axis_0_label_to_index[axis_0_label]

        axis_1_index = axis_1_label_to_index[axis_1_label]

        value_now = matrix[axis_0_index, axis_1_index]

        if isnan(value_now) or function is None:

            matrix[axis_0_index, axis_1_index] = value

        else:

            matrix[axis_0_index, axis_1_index] = function(value_now, value)

    return DataFrame(
        data=matrix,
        index=Index(data=axis_0_label_to_index, name=axis_0_name),
        columns=Index(data=axis_1_label_to_index, name=axis_1_name),
    )


def binarize(sr):

    value_to_axis_0_index = {}

    axis_0_index = 0

    for value in sr:

        if not isna(value) and value not in value_to_axis_0_index:

            value_to_axis_0_index[value] = axis_0_index

            axis_0_index += 1

    matrix = full((len(value_to_axis_0_index), sr.size), 0)

    for (axis_1_index, value) in enumerate(sr):

        if not isna(value):

            matrix[value_to_axis_0_index[value], axis_1_index] = 1

    return DataFrame(
        data=matrix,
        index=Index(data=value_to_axis_0_index, name=sr.name),
        columns=sr.index,
    )
