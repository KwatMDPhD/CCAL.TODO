def _process_target(tav, ty, st):
    if ty == "continuous":
        if 0 < tav.std():
            tav = apply(tav, normalize, "-0-", up=True).clip(min=-st, max=st)

        return tav, -st, st

    else:
        return tav.copy(), None, None


def _process_target_and_get_1(tav, ty, st):
    return _process_target(tav, ty, st)[0]


def _process_data(dav, ty, st):
    dav = apply_along_axis(_process_target_and_get_1, 1, dav, ty, st)

    if ty == "continuous":
        return dav, -st, st

    else:
        return dav, None, None


ANNOTATION = {
    "xref": "paper",
    "yref": "paper",
    "yanchor": "middle",
    "font": {"size": 10},
    "showarrow": False,
}

HEATMAP = {"type": "heatmap", "showscale": False}

LAYOUT = {"width": 800, "margin": {"l": 200, "r": 200}, "title": {"x": 0.5}}


def _make_target_annotation(y, ro):
    return [
        merge(
            ANNOTATION,
            {"y": y, "x": 0, "xanchor": "right", "text": "<b>{}</b>".format(trim(ro))},
        )
    ]


def _get_x(ie):
    return 1.08 + ie / 6.4


def _make_data_annotation(y, layout, he, ro_, st):
    annotations = []

    if la:
        for ie, text in enumerate(["Score (\u0394)", "P-Value", "Q-Value"]):
            annotations.append(
                merge(
                    ANNOTATION,
                    {
                        "y": y,
                        "x": _get_x(ie),
                        "xanchor": "center",
                        "text": "<b>{}</b>".format(text),
                    },
                )
            )

    y -= he

    for iey in range(ro_.size):
        annotations.append(
            merge(
                ANNOTATION,
                {
                    "y": y,
                    "x": 0,
                    "xanchor": "right",
                    "text": "{}".format(trim(ro_[iey])),
                },
            )
        )

        sc, ma, pv, qv = st[iey]

        for iex, text in enumerate(
            ["{:.2f} ({:.2f})".format(sc, ma), "{:.2e}".format(pv), "{:.2e}".format(qv)]
        ):
            annotations.append(
                merge(
                    ANNOTATION,
                    {"y": y, "x": _get_x(iex), "xanchor": "center", "text": text},
                )
            )

        y -= he

    return annotations
