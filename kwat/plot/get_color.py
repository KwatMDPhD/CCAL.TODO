from plotly.colors import convert_colors_to_same_type, find_intermediate_color


def get_color(colorscale, fr):

    for ie in range(len(colorscale) - 1):

        fr1, co1 = colorscale[ie]

        fr2, co2 = colorscale[ie + 1]

        if fr1 <= fr <= fr2:

            return find_intermediate_color(
                *convert_colors_to_same_type([co1, co2])[0],
                (fr - fr1) / (fr2 - fr1),
                colortype="rgb",
            )
