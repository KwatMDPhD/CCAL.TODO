from .RANDOM_SEED import RANDOM_SEED


def sample_from_each_series_value(series, n_per_value=None, random_seed=RANDOM_SEED):

    if n_per_value is None:

        n_per_value = series.value_counts().min()

        print("n_per_value = {}".format(n_per_value))

    indices_selected = []

    for group_name, group_series in series.groupby(by=series):

        if n_per_value <= group_series.size:

            indices_selected.extend(
                group_series.sample(
                    n=n_per_value, random_state=random_seed
                ).index.sort_values()
            )

        else:

            print(
                "Not sampling {}, which appears less than {} times.".format(
                    group_name, n_per_value
                )
            )

    return series[indices_selected]
