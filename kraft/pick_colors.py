from matplotlib.colors import to_hex
from numpy import asarray, unique
from seaborn import husl_palette

from .check_array_for_bad import check_array_for_bad
from .get_colormap_colors import get_colormap_colors
from .get_data_type import get_data_type


def pick_colors(data, data_type=None):

    if data_type is None:

        data_type = get_data_type(data)

    if data_type == "binary":

        return ("#ebf6f7", "#171412")

    elif data_type == "categorical":

        data = asarray(data)

        return tuple(
            to_hex(rgb)
            for rgb in husl_palette(
                n_colors=unique(
                    data[~check_array_for_bad(data, raise_for_bad=False)]
                ).size
            )
        )

    elif data_type == "continuous":

        return get_colormap_colors("bwr")
