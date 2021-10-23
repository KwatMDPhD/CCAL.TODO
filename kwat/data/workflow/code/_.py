import os  # isort: skip
import re  # isort: skip

import numpy as np  # isort: skip
import pandas as pd  # isort: skip
import plotly as pl  # isort: skip

import kwat  # isort: skip

pas = os.path.join("..", "input", "setting.json")

PAR, PAI, PAC, PAO = get_path(pas)

SE = read_setting(pas)

# ==============================================================================
