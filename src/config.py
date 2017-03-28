"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

import os

# Set global paths for data, results, and wts directories.
DATA_PATH    = '../data'
WTS_PATH     = '../wts'
RESULTS_PATH = '../results'


# Create dirs on import, if needed.
if not os.path.exists(DATA_PATH):
    os.makedirs(DATA_PATH)

if not os.path.exists(WTS_PATH):
    os.makedirs(WTS_PATH)

if not os.path.exists(RESULTS_PATH):
    os.makedirs(RESULTS_PATH)
