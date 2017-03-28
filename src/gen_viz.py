"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Generates some visualizations from test results
# Arguments:
# map.txt - map file
# expname - name of experiment file
# outdir - where to save the results

import sys
import os
import numpy as np
import viz
from config import *

if len(sys.argv) != 2:
    sys.exit('USAGE: gen_viz.py expname')

exp = sys.argv[1]
results_dir = os.path.join(RESULTS_PATH, exp)

assert os.path.isfile(results_dir + '/results.npz'), "Run eval_model.py before generating visualizations."

viz.draw_map(exp, os.path.join(results_dir, 'map.png'))
viz.draw_heatmap(exp, os.path.join(results_dir, 'error.png'))
