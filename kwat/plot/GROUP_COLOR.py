from ..constant import NUMBER_OF_CATEGORY
from .CATEGORICAL_COLORSCALE import CATEGORICAL_COLORSCALE
from .get_color import get_color

GROUP_COLOR = {}

for ie in range(NUMBER_OF_CATEGORY):

    GROUP_COLOR[ie + 1] = get_color(CATEGORICAL_COLORSCALE, ie / n_co)
