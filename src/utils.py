"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

import re
import os
from glob import glob
import numpy as np
import tensorflow as tf

# Reading in batches
# repeatable random shuffling
class batcher:
    def __init__(self,d,bsz,niter=0,shuffle=True):

        self.data = d
        self.bsz = bsz
        self.shuffle = shuffle

        self.rand = np.random.RandomState(0)

        if self.shuffle:
            idx = self.rand.permutation(d[0].shape[0])
            for i in range(niter*bsz // len(idx)):
                idx = self.rand.permutation(len(idx))
        else:
            idx = range(d[0].shape[0])

        self.idx = np.int32(idx)
        self.pos = niter*bsz % len(self.idx)

    def get_batch(self):
        if self.pos+self.bsz >= len(self.idx):
            bidx = self.idx[self.pos:]

            if self.shuffle:
                idx = self.rand.permutation(len(self.idx))
                self.idx = np.int32(idx)

            self.pos = 0
            if len(bidx) < self.bsz:
                self.pos = self.bsz-len(bidx)
                bidx2 = self.idx[0:self.pos]
                bidx = np.concatenate((bidx,bidx2))
        else:
            bidx = self.idx[self.pos:self.pos+self.bsz]
            self.pos = self.pos+self.bsz

        return [d[bidx,...] for d in self.data]

# Manage checkpoint files, read off iteration number from filename
# Use clean() to keep latest, and modulo n iters, delete rest
class ckpter:
    def __init__(self,wcard):
        self.wcard = wcard
        self.load()

    def load(self):
        lst = glob(self.wcard)
        if len(lst) > 0:
            lst=[(l,int(re.match('.*/.*_(\d+)',l).group(1)))
                 for l in lst]
            self.lst=sorted(lst,key=lambda x: x[1])

            self.iter = self.lst[-1][1]
            self.latest = self.lst[-1][0]
        else:
            self.lst=[]
            self.iter=0
            self.latest=None

    def clean(self,every=0,last=1):
        self.load()
        old = self.lst[:-last]
        for j in old:
            if every == 0 or j[1] % every != 0:
                os.remove(j[0])

## Reading and saving weights

def netload(net,fname,sess):
    wts = np.load(fname)
    for k in wts.keys():
        wvar = net.weights[k]
        wk = wts[k].reshape(wvar.get_shape())
        sess.run(wvar.assign(wk))

# Save weights to an npz file
def netsave(net,fname,sess):
    wts = {}
    for k in net.weights.keys():
        wts[k] = net.weights[k].eval(sess)
    np.savez(fname,**wts)
