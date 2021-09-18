from plotly.io import show, write_html

from ..dictionary import merge


def plot_plotly(figure, pa=""):

    figure = merge(
        {
            "layout": {
                "autosize": False,
                "template": "plotly_white",
            },
        },
        figure,
    )

    di = {
        "editable": True,
    }

    show(figure, config=di)

    if pa != "":

        write_html(figure, pa, config=di)
