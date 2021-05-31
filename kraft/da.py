def map_to_column(da, co, fu=None):

    ke_va = {}

    for va, ke_ in zip(da.pop(co).to_numpy(), da.to_numpy()):

        for ke in ke_:

            if fu is None:

                ke_va[ke] = va

            else:

                for ke in fu(ke):

                    ke_va[ke] = va

    return ke_va


def error_axis(da):

    for la_ in da.axes:

        assert not la_.isna().any(), "Has Na."

        assert not la_.has_duplicates, "Duplicated."


def count(da):

    for la, an_ in da.iteritems():

        print("-" * 80)

        print(la)

        an_co = an_.value_counts()

        print(an_co)

        print("-" * 80)


def drop_axis_label(df, axis, not_na_min_n=None, not_na_unique_min_n=None):

    assert not_na_min_n is not None and not_na_unique_min_n is not None

    shape_before = df.shape

    is_keep_ = full(shape_before[axis], True)

    if axis == 0:

        apply_axis = 1

    elif axis == 1:

        apply_axis = 0

    matrix = df.to_numpy()

    if not_na_min_n is not None:

        if not_na_min_n < 1:

            not_na_min_n = not_na_min_n * shape_before[apply_axis]

        is_keep_ &= apply_along_axis(
            _check_has_enough_not_na, apply_axis, matrix, not_na_min_n
        )

    if not_na_unique_min_n is not None:

        if not_na_unique_min_n < 1:

            not_na_unique_min_n = not_na_unique_min_n * df.shape[apply_axis]

        is_keep_ &= apply_along_axis(
            _check_has_enough_not_na_unique, apply_axis, matrix, not_na_unique_min_n
        )

    if axis == 0:

        df = df.loc[is_keep_, :]

    elif axis == 1:

        df = df.loc[:, is_keep_]

    print("{} => {}".format(shape_before, df.shape))

    return df


def drop_axes_label(df, axis=None, not_na_min_n=None, not_na_unique_min_n=None):

    shape_before = df.shape

    if axis is None:

        axis = int(shape_before[0] < shape_before[1])

    can_return = False

    while True:

        df = drop_axis_label(
            df, axis, not_na_min_n=not_na_min_n, not_na_unique_min_n=not_na_unique_min_n
        )

        shape_after = df.shape

        if can_return and shape_before == shape_after:

            return df

        shape_before = shape_after

        if axis == 0:

            axis = 1

        elif axis == 1:

            axis = 0

        can_return = True
