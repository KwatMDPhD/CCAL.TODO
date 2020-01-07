from .flatten_nested_iterable import flatten_nested_iterable


def make_consecutive_group_labels(n, n_group):

    n_per_group = n // n_group

    labels = flatten_nested_iterable(((i,) * n_per_group for i in range(n_group)))

    if len(labels) < n:

        labels += labels[-1:]

    return labels
