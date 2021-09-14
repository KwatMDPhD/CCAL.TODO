from plotly.io import show, write_html

from ..dictionary import merge


def plot_plotly(figure, pa=""):

    figure = merge(
        {
            "LAYOUT_TEMPLATE": {
                "autosize": False,
                "template": "plotly_white",
            },
        },
        figure,
    )

    co = {
        "editable": True,
    }

    show(figure, config=co)

    if pa != "":

        write_html(figure, pa, config=co)
