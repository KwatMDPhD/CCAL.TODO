from plotly.colors import convert_colors_to_same_type, find_intermediate_color


def get_colorscale_color(colorscale, value):

    assert 0 <= value <= 1

    for i in range(len(colorscale) - 1):

        if colorscale[i][0] <= value <= colorscale[i + 1][0]:

            scale_low, color_low = colorscale[i]

            scale_high, color_high = colorscale[i + 1]

            value_ = (value - scale_low) / (scale_high - scale_low)

            color = find_intermediate_color(
                *convert_colors_to_same_type((color_low, color_high))[0],
                value_,
                colortype="rgb"
            )

            return "rgb{}".format(
                tuple(int(float(i)) for i in color[4:-1].split(sep=","))
            )
