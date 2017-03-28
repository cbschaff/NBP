"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

import sys
import os
import time
import numpy as np
from importlib import import_module
import matplotlib.pyplot as plt
import utils as ut
from plan import plan
from config import *

# Visualizes a map with beacon locations and channels
# Args:
# modelfile - file containing model parameters (used to determine beacon placement)
# outfile - where to save the plot. (shows a figure if None)
# it - iteration number of the model file, shown in plot title.
def draw_map(exp, outfile=None, it=None):

    p = import_module('experiments.{}'.format(exp))
    pl = plan(p.MAPFILE)
    nTX = pl.TXs.shape[0]

    wts_dir = os.path.join(WTS_PATH, exp)
    if it == None:
        saved = ut.ckpter(os.path.join(wts_dir, 'iter_*.model.npz'))
        modelfile = saved.latest
    else:
        modelfile = os.path.join(wts_dir, 'iter_{}.model.npz'.format(it))

    # Load model and get beacon layout
    tx = np.load(modelfile)
    nChan = tx['sensor'].shape[0]//nTX-1
    tx = np.argmax(tx['sensor'].reshape([nTX,-1]),axis=1)

    plt.clf()
    plt.hold(True)

    # Draw walls
    for j in range(pl.walls.shape[0]):
        wj = pl.walls[j,...]
        plt.plot([wj[0],wj[2]],[wj[1],wj[3]],'-k')

    # Get color code
    clrs = []
    for j in range(nChan):
        clr = plt.cm.jet( int(round(float(j)/float(nChan-1)*255.)) )
        clr = '#' + ''.join(['%02x' % round(d*255) for d in clr[0:3]])
        clrs.append(clr)

    # Draw transmitters
    for j in range(nTX):
        if tx[j] > 0:
            plt.plot(pl.TXs[j,0],pl.TXs[j,1],'.',color=clrs[tx[j]-1])

    plt.xlim([0,pl.w])
    plt.ylim([0,pl.h])
    plt.xticks([])
    plt.yticks([])
    if it is not None:
        plt.title('Iteration: {}'.format(it))

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)

# Visualizes a heatmap of errors at test locations
# Args:
# exp - experiment file
# test_results - dictionary containing results from eval_model.py
# outfile - where to save the plot. (shows a figure if None)
# stat - the statistic to show on the heatmap. Options are : ['mean', 'median', 'max', 'min']
def draw_heatmap(exp, outfile=None, stat='mean'):

    assert stat in ['mean', 'median', 'max', 'min'], "Invalid Statistic"

    results_file = os.path.join(RESULTS_PATH, exp, 'results.npz')
    test_results = np.load(results_file)
    locs = test_results['xy']
    preds = test_results['preds']

    p = import_module('experiments.' + exp)
    pl = plan(p.MAPFILE)

    nTX = pl.TXs.shape[0]

    # Load model and get beacon layout
    saved = ut.ckpter(os.path.join(WTS_PATH, exp, 'iter_*.model.npz'))
    tx = np.load(saved.latest)
    nChan = tx['sensor'].shape[0]//nTX-1
    tx = np.argmax(tx['sensor'].reshape([nTX,-1]),axis=1)

    plt.figure()
    plt.hold(True)

    # Draw walls
    for j in range(pl.walls.shape[0]):
        wj = pl.walls[j,...]
        plt.plot([wj[0],wj[2]],[wj[1],wj[3]],'-k')

    # Get color code
    clrs = []
    for j in range(nChan):
        clr = plt.cm.jet( int(round(float(j)/float(nChan-1)*255.)) )
        clr = '#' + ''.join(['%02x' % round(d*255) for d in clr[0:3]])
        clrs.append(clr)

    # Draw transmitters
    for j in range(nTX):
        if tx[j] > 0:
            plt.plot(pl.TXs[j,0],pl.TXs[j,1],'.',color=clrs[tx[j]-1])

    # Compute heatmap
    if stat == 'mean':
        heatmap = np.sqrt(np.mean(np.sum((preds - locs[:,np.newaxis,:])**2, axis=2), axis=1))
    if stat == 'median':
        heatmap = np.sqrt(np.median(np.sum((preds - locs[:,np.newaxis,:])**2, axis=2), axis=1))
    if stat == 'min':
        heatmap = np.sqrt(np.min(np.sum((preds - locs[:,np.newaxis,:])**2, axis=2), axis=1))
    if stat == 'max':
        heatmap = np.sqrt(np.max(np.sum((preds - locs[:,np.newaxis,:])**2, axis=2), axis=1))


    # Draw errors
    extent = [0,pl.w,0,pl.h]
    unx = np.unique(locs[:,0])
    uny = np.unique(locs[:,1])

    # shift bin boundaries
    xsize = unx[1] - unx[0]
    ysize = uny[1] - uny[0]
    x_bins = np.concatenate((unx - (xsize / 2.), unx[-1:] + (xsize / 2.)))
    y_bins = np.concatenate((uny - (ysize / 2.), uny[-1:] + (ysize / 2.)))
    hist,_,_ = np.histogram2d(locs[:,1], locs[:,0], bins=(y_bins, x_bins), weights=heatmap)

    plt.imshow(hist, extent=extent, origin='lower', vmin=0.0, vmax=0.5)
    plt.colorbar(fraction = 0.04)

    plt.xlim([0,pl.w])
    plt.ylim([0,pl.h])
    plt.xticks([])
    plt.yticks([])
    plt.title(stat)

    if outfile is None:
        plt.show()
    else:
        plt.savefig(outfile)
