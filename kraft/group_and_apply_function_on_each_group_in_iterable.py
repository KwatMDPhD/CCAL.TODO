from numpy import asarray
from pandas import unique


def group_and_apply_function_on_each_group_in_iterable(iterable, groups, function):

    unique_groups_in_order = unique(groups)

    applied_by_group = []

    for group in unique_groups_in_order:

        applied_by_group.append(function(asarray(iterable)[groups == group]))

    return unique_groups_in_order, applied_by_group
