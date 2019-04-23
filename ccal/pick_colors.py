from matplotlib.colors import to_hex
from numpy import asarray, isnan, unique
from seaborn import husl_palette

from .get_colormap_colors import get_colormap_colors
from .get_data_type import get_data_type


def pick_colors(data):

    data_type = get_data_type(data)

    if data_type == "binary":

        return ("#ebf6f7", "#171412")

    elif data_type == "categorical":

        data = asarray(data)

        n_color = unique(data[~isnan(data)]).size

        curated_colors = (
            "#20d9ba",
            "#9017e6",
            "#ff1968",
            "#ffe119",
            "#3cb44b",
            "#4e41d8",
            "#ffa400",
            "#aaffc3",
            "#800000",
            "#e6beff",
            "#fffac8",
            "#0082c8",
            "#e6194b",
            "#006442",
            "#46f0f0",
            "#bda928",
            "#c91f37",
            "#fabebe",
            "#d2f53c",
            "#aa6e28",
            "#ff0000",
            "#808000",
            "#003171",
            "#ff4e20",
            "#a4345d",
            "#ffd8b1",
            "#bb7796",
            "#f032e6",
        )

        if n_color <= len(curated_colors):

            return curated_colors[:n_color]

        else:

            return tuple(to_hex(rgb) for rgb in husl_palette(n_colors=n_color))

    elif data_type == "continuous":

        return get_colormap_colors("bwr")
