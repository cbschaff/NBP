"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Training script. Trains a model with hyperparams defined in a .py file
# in the experiments folder.
# USAGE: run.py expname,
# where experiments/expname.py is a file.

import sys
import os
import time
import tensorflow as tf
import numpy as np
import utils as ut
from importlib import import_module
from trainer import train
from config import *

def mprint(s):
    sys.stdout.write(s+"\n")
    sys.stdout.flush()



# Get experiments and parameters from a .py file
# specified by the command line.
if len(sys.argv) != 2:
    sys.exit("USAGE: run.py expname, where experiments/expname.py is a file.")
exp = sys.argv[1]
p = import_module('experiments.' + exp)

# Load training set.
mapname = os.path.basename(p.MAPFILE).split('.')[0]
f = np.load(os.path.join(DATA_PATH, '{}_train.npz'.format(mapname)))
NTX = f['dist'].shape[1]

# Create wtsdir if it doesn't exist
wts_dir = os.path.join(WTS_PATH, exp)
if not os.path.exists(wts_dir):
    os.makedirs(wts_dir)
# Check for saved weights & find iter
saved = ut.ckpter(os.path.join(wts_dir, 'iter_*.model.npz'))
iter = saved.iter

logfile = os.path.join(wts_dir, 'log.npz')
if os.path.exists(logfile):
    npzf = np.load(logfile)
    log = dict([(k,npzf[k]) for k in npzf])
else:
    log = {'iter':np.zeros((0,)),
        'loss':np.zeros((0,)),
        'entropy':np.zeros((0,)),
        'hard-iter':np.zeros((0,)),
        'hard-loss':np.zeros((0,))}

# Set up random batches from training set
# seeking to iter.
batcher = ut.batcher([f['xy'], f['dist'], f['nint']], p.BSZ, iter)

# Create placeholders for input and GT location
dists = tf.placeholder(shape=(p.BSZ,NTX),dtype=tf.float32)
nints = tf.placeholder(shape=(p.BSZ,NTX),dtype=tf.float32)
gtlocs = tf.placeholder(shape=(p.BSZ,2),dtype=tf.float32)

itr = tf.placeholder(shape=(),dtype=tf.int32)

# Load model
net = p.md.model(NTX,dists,nints,itr)

# Load optimizer
if 'cost' not in net.__dict__:
    net.cost = None

# Check for beacon regularization annealing
beacr = p.BEACON_DECAY
if hasattr(p, 'BEACON_ANNEAL') and hasattr(p, 'BEACON_ANNEAL_FREQ'):
    beacr *= tf.pow(p.BEACON_ANNEAL, tf.to_float(itr / p.BEACON_ANNEAL_FREQ))
lossT, tstep = train(net, gtlocs, p.LR, p.MOM, p.WEIGHT_DECAY, beacr)


tsteps = [tstep]
if 'bnops' in net.__dict__:
    tsteps = tsteps+net.bnops

# Start TF session (respecting cluster num_threads_var)
nthr = os.getenv('OMP_NUM_THREADS')
if nthr is None:
    sess = tf.Session()
else:
    sess = tf.Session(config=tf.ConfigProto(
        intra_op_parallelism_threads=int(nthr)))

# Init everything
try: # tensorflow 1.0
    sess.run(tf.global_variables_initializer())
except: # tensorflow < 1.0
    sess.run(tf.initialize_all_variables())

# Timestamp display format
stamp = lambda: time.strftime("%Y-%m-%d %H:%M:%S ")

# Load saved weights if any
if saved.latest is not None:
    mprint(stamp()+"Restoring from " + saved.latest )
    ut.netload(net,saved.latest,sess)
    saved.clean(every=p.KEEPEVERY,last=p.KEEPLAST)

# Training loop
s_loss = []
s_ent = []
stop=False
mprint(stamp()+"Starting from Iteration %d" % iter)
try:
    while iter < p.MAX_ITER and not stop:

        # Run training step & get next batch
        bdata = batcher.get_batch()
        fdict = {
            gtlocs: bdata[0],
            dists: bdata[1],
            nints: bdata[2],
            itr: iter,

        }
        outs = sess.run([lossT, net.entropy] + tsteps,
                        feed_dict=fdict)
        s_loss.append(outs[0])
        s_ent.append(outs[1])

        # Hard-decision loss
        if p.HARD_FREQ > 0 and iter % p.HARD_FREQ == 0:
            fdict[itr] = -1
            [loss,rxp] = sess.run([lossT,net.rxpow],feed_dict=fdict)
            mprint(stamp() + "[%09d] Hard loss = %.6f"
                             % (iter,loss))

            rxp = np.mean(rxp,axis=0)
            rxp = ",".join(["%.2f" % rxp[j] for j in range(rxp.shape[0])])
            mprint(stamp() + ("[%09d] Avg Pow = " % iter) + rxp)

            if iter != 0:
                log['hard-loss'] = np.append(log['hard-loss'], [loss])
                log['hard-iter'] = np.append(log['hard-iter'], [iter])

        ## Display
        if iter % p.DISP_FREQ == 0:
            loss = np.mean(s_loss)
            entropy = np.mean(s_ent)
            s_loss, s_ent = [], []
            mprint(stamp() + "[%09d] lr=%.2e Train loss = %.6f Beacon entropy = %.6f"
                             % (iter,p.LR,loss,entropy))

            # record loss and entropy
            if iter != 0:
                log['iter'] = np.append(log['iter'], [iter])
                log['loss'] = np.append(log['loss'], [loss])
                log['entropy'] = np.append(log['entropy'], [entropy])
        iter=iter+1

        ## Save
        if p.SAVE_FREQ > 0 and iter % p.SAVE_FREQ == 0:
            fname = os.path.join(wts_dir, "iter_%d.model.npz" % iter)
            ut.netsave(net,fname,sess)
            np.savez(logfile, **log)
            saved.clean(every=p.KEEPEVERY,last=p.KEEPLAST)
            mprint(stamp() + "Saved weights to " + fname )

except KeyboardInterrupt: # Catch ctrl+c/SIGINT
    mprint(stamp()+"Stopped!")
    stop = True
    pass

# Save last
if saved.iter < iter:
    fname = os.path.join(wts_dir, "iter_%d.model.npz" % iter)
    ut.netsave(net,fname,sess)
    np.savez(logfile, **log)
    saved.clean(every=p.KEEPEVERY,last=p.KEEPLAST)
    mprint(stamp()+"Saved weights to " + fname )
