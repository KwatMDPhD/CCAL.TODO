from plotly.io import show, templates, write_html

templates["kraft"] = {"layout": {"autosize": False}}


def plot_plotly_figure(figure, html_file_path=None):

    template = "plotly_white+kraft"

    if "layout" in figure:

        figure["layout"]["template"] = template

    else:

        figure["layout"] = {"template": template}

    config = {"editable": False}

    show(figure, config=config)

    if html_file_path is not None:

        write_html(figure, html_file_path, config=config)
