"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Generates training data for a given map. The data consists of tuples
# of the form:
#     - a random (x,y) location in the map
#     - the distance to each beacon from this location
#     - the number of interfering walls between each beacon and this location

from __future__ import print_function
import plan as pl
import numpy as np
import sys
import os
import time
from config import *

BSZ = 500 # batch size
LOCFAC = 1000 # Divide co-ordinates by this for stability

def gen_train_data(mapfile, nbatch):
    # data is saved in DATA_PATH/mapname_train.npz
    mapname = os.path.basename(mapfile).split('.')[0]
    out = os.path.join(DATA_PATH, '{}_train.npz'.format(mapname))

    p = pl.plan(mapfile)

    xy = np.zeros((BSZ*nbatch,2), dtype=np.float32)
    dist = np.zeros((BSZ*nbatch, p.TXs.shape[0]), dtype=np.float32)
    nint = np.zeros((BSZ*nbatch, p.TXs.shape[0]), dtype=np.int32)

    np.random.seed(int(time.time()))

    for i in range(nbatch):
        xyi = np.random.random((BSZ,2))*[[p.w,p.h]]
        di,ni = p.getDX(xyi)
        xy[i*BSZ:(i+1)*BSZ,...] = xyi
        dist[i*BSZ:(i+1)*BSZ,...] = di
        nint[i*BSZ:(i+1)*BSZ,...] = ni

        print('Generated %d of %d [x%d]' % (i+1,nbatch,BSZ))

    sdict = {'xy': xy / LOCFAC, \
             'dist': dist / LOCFAC, \
             'nint': nint}
    np.savez(out, **sdict)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        sys.exit('Usage: genData.py map.txt [nbatch]')

    mapfile = sys.argv[1]
    if len(sys.argv) < 3:
        nbatch = 2000
    else:
        nbatch = int(sys.argv[2])

    gen_train_data(mapfile, nbatch)
