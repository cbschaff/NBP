"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# This experiment learns to localize with our best performing fixed
# beacon layout on Map 2.

import models.inf as md
import models.beacon14 as smod
import models.prop1 as pmod

md.smod = smod
md.pmod = pmod

# Number of layer blocks in the neural net
md.nGroup = 6


# Map file.
MAPFILE = '../maps/map2.txt'

# Determine which checkpoints to keep
KEEPLAST = 1
KEEPEVERY= 1e6
SAVE_FREQ= 10000

# How often to display loss compute loss under a hard beacon assignment
DISP_FREQ=25
HARD_FREQ=500

# Optimization hyperparams
MOM = 0.9
WEIGHT_DECAY=0.
BEACON_DECAY=0.
BSZ=1000

LR = 0.01
MAX_ITER = 1e6
