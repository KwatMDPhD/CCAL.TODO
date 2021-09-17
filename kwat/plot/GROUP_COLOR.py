from .CATEGORICAL_COLORSCALE import CATEGORICAL_COLORSCALE
from .get_color import get_color

GROUP_COLOR = {}

n_co = 24

for ie in range(n_co):

    GROUP_COLOR[ie + 1] = get_color(CATEGORICAL_COLORSCALE, ie / n_co)
