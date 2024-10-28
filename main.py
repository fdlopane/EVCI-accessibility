"""
Measuring supply and demand of electric vehicle charging infrastructure (EVCI) in London

Author Fulvio D. Lopane
Centre for Advanced Spatial Analysis

started coding: October 2024
"""

import pandas as pd
from config import *


# Import EVCI counts at lsoa level
EVCI = pd.read_csv(inputs["EVCI-LSOA"])

# Import EV licensing counts at lsoa level
EV_counts = pd. read_csv(inputs["EV-counts-LSOA"])

print(EVCI)
print(EV_counts)