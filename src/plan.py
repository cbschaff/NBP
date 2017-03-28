"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Computes possible beacon locations, distances, and number of interfering walls.

from __future__ import division
import numpy as np

_eps=1e-4

class plan:
    def __init__(self,fname):
        # Parse file
        d = open(fname,'rt').readlines()
        d = [[float(n) for n in l.rstrip("\n").split(" ")] for l in d]

        assert len(d[0]) == 2, \
            "First line should have two entries for room width, height."
        assert len(d[1]) == 2, \
            "Second line should have two entries for Tx_nHoriz, Tx_nVert."

        # Width height
        self.w, self.h = d[0]

        # Regular grid of transmitters
        tx, ty = d[1]
        tx,ty = np.meshgrid(
            np.linspace(0.,self.w,int(tx+2))[1:-1],
            np.linspace(0.,self.h,int(ty+2))[1:-1])
        self.TXs = np.stack([tx.flatten(),ty.flatten()],axis=1)

        # Parse wall co-ordinates
        walls = []
        for j in range(2,len(d)):
            dj = d[j]
            assert len(dj) == 4, \
                "Wall description lines should have four numbers (x1,y1,x2,y2)."
            walls.append(np.float32(dj))
        walls = np.stack(walls)
        self.walls = walls

        # Store as lines
        self.wa = walls[:,3]-walls[:,1]
        self.wb = -walls[:,2]+walls[:,0]
        self.wc = walls[:,0]*self.wa + walls[:,1]*self.wb

        self.xmin = np.min(self.walls[:,0::2],axis=1)
        self.ymin = np.min(self.walls[:,1::2],axis=1)
        self.xmax = np.max(self.walls[:,0::2],axis=1)
        self.ymax = np.max(self.walls[:,1::2],axis=1)

    # Get distance and number of crossings
    # rloc is Nx2
    # returns d, nc : N x nTx
    def getDX(self,rloc):
        nax = np.newaxis

        # Compute distance
        dxy = rloc[:,:,nax]-self.TXs.transpose()[nax,:,:]
        d = np.sqrt(np.sum(dxy**2,axis=1))

        # No. of walls intersected

        # Line eq from Tx to R
        la = rloc[:,1,nax] - self.TXs.transpose()[nax,1,:]
        lb = -rloc[:,0,nax] + self.TXs.transpose()[nax,0,:]
        lc = (rloc[:,0,nax]*la+rloc[:,1,nax]*lb)

        # Find intersections
        det=la[:,:,nax]*self.wb[nax,nax,:] - \
             lb[:,:,nax]*self.wa[nax,nax,:]
        xint = (self.wb[nax,nax,:]*lc[:,:,nax] - \
                lb[:,:,nax]*self.wc[nax,nax,:])/det
        yint = (-self.wa[nax,nax,:]*lc[:,:,nax] + \
                la[:,:,nax]*self.wc[nax,nax,:])/det

        # Check if intersection is within segments
        xmin = np.minimum(rloc[:,0,nax],self.TXs[nax,:,0])
        xmax = np.maximum(rloc[:,0,nax],self.TXs[nax,:,0])
        ymin = np.minimum(rloc[:,1,nax],self.TXs[nax,:,1])
        ymax = np.maximum(rloc[:,1,nax],self.TXs[nax,:,1])

        nint = np.ones(xint.shape,dtype=np.int32)
        nint = nint * np.int32(xint >= xmin[:,:,nax]-_eps)
        nint = nint * np.int32(yint >= ymin[:,:,nax]-_eps)
        nint = nint * np.int32(xint <= xmax[:,:,nax]+_eps)
        nint = nint * np.int32(yint <= ymax[:,:,nax]+_eps)

        nint = nint * np.int32(xint >= self.xmin[nax,nax,:]-_eps)
        nint = nint * np.int32(yint >= self.ymin[nax,nax,:]-_eps)
        nint = nint * np.int32(xint <= self.xmax[nax,nax,:]+_eps)
        nint = nint * np.int32(yint <= self.ymax[nax,nax,:]+_eps)

        return d, np.sum(nint,axis=2)
