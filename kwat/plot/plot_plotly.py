from plotly.io import show, write_html, write_image

from ..dictionary import merge


def plot_plotly(figure, pa=""):

    axis = {"automargin": True}

    figure = merge(
        {
            "layout": {
                "autosize": False,
                "template": "plotly_white",
                "xaxis": axis,
                "yaxis": axis,
            }
        },
        figure,
    )

    config = {
        "editable": False,
        "modeBarButtonsToAdd": ["drawopenpath", "eraseshape"],
        "displaylogo": False,
    }

    show(figure, config=config)

    if pa != "":

        write_html(figure, pa, config=config)

        write_image(figure, pa.replace(".html", ".png"))
