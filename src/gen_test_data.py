"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Generates test data for a given map. The data consists of tuples
# of the form:
#     - a (x,y) location in the map
#     - the distance to each beacon from this location
#     - the number of interfering walls between each beacon and this location
# where each location is generated on a dense grid covering the map.

from __future__ import print_function
import plan as pl
import numpy as np
import sys
import os
import time
from config import *


BSZ = 500 # batch size
SKIP = 4 # number of locations to skip between test points.
LOCFAC = 1000 # Divide co-ordinates by this for stability

def gen_test_data(mapfile):
    # data is saved in DATA_PATH/mapname_test.npz
    mapname = os.path.basename(mapfile).split('.')[0]
    out = os.path.join(DATA_PATH, '{}_test.npz'.format(mapname))

    p = pl.plan(mapfile)

    print("Generating Test Data")
    xv,yv = np.meshgrid(range(1, int(p.w), SKIP), range(1, int(p.h), SKIP))
    xy = np.float32(np.reshape((xv,yv), (2,-1)).T)

    n = xy.shape[0]
    dist = np.zeros((n, p.TXs.shape[0]), dtype=np.float32)
    nint = np.zeros((n, p.TXs.shape[0]), dtype=np.int32)

    pos = 0
    while pos < n:
        end = pos + BSZ if pos + BSZ <= n else n
        xyi = xy[pos:end,...]
        di,ni = p.getDX(xyi)
        dist[pos:end,...] = di
        nint[pos:end,...] = ni
        pos += BSZ
        print('Generated %d of %d [x%d]' % (pos / BSZ, (n // BSZ) + 1, BSZ))

    sdict = {'xy': xy / LOCFAC, \
             'dist': dist / LOCFAC, \
             'nint': nint}

    np.savez(out, **sdict)


if __name__ == '__main__':

    if len(sys.argv) != 2:
        sys.exit('genData.py map.txt')

    mapfile = sys.argv[1]
    gen_test_data(mapfile)
