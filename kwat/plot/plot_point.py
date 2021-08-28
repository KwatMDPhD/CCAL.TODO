from .plot_plotly import plot_plotly


def plot_point(an_po_pa, title="", pa=""):

    data = [
        {
            "name": "Point",
            "y": an_po_pa.iloc[:, 0],
            "x": an_po_pa.iloc[:, 1],
            "text": an_po_pa.index.values,
            "mode": "markers",
            "marker": {
                "size": an_po_pa.loc[:, "Size"],
                "color": an_po_pa.loc[:, "Color"],
                "opacity": an_po_pa.loc[:, "Opacity"],
                "line": {
                    "width": 0,
                },
            },
        }
    ]

    co_ = an_po_pa.columns.values

    annotations = []

    if "Annotate" in co_:

        for text, (y, x) in (
            an_po_pa.iloc[:, :2].loc[an_po_pa.loc[:, "Annotate"], :].iterrows()
        ):

            annotations.append(
                {
                    "y": y,
                    "x": x,
                    "text": text,
                    "font": {
                        "size": 8,
                    },
                    "arrowhead": 2,
                    "arrowsize": 1,
                    "clicktoshow": "onoff",
                }
            )

    plot_plotly(
        {
            "data": data,
            "layout": {
                "title": title,
                "yaxis": {
                    "title": co_[0],
                },
                "xaxis": {
                    "title": co_[1],
                },
                "annotations": annotations,
            },
        },
        pa=pa,
    )
