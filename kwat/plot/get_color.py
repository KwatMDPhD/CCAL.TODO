from plotly.colors import convert_colors_to_same_type, find_intermediate_color


def _scale(mi, nu, ma):

    return (nu - mi) / (ma - mi)


def get_color(colorscale, nu, ex_=()):

    if len(ex_) == 2:

        mi, ma = ex_

        nu = _scale(mi, nu, ma)

    for ie in range(len(colorscale) - 1):

        fr1, co1 = colorscale[ie]

        fr2, co2 = colorscale[ie + 1]

        if fr1 <= nu <= fr2:

            return find_intermediate_color(
                *convert_colors_to_same_type([co1, co2])[0],
                _scale(fr1, nu, fr2),
                colortype="rgb",
            )
