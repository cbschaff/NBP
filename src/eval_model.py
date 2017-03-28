"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Evaluates a trained model on a test set
# USAGE: eval_model.py expname test_data.npz nsample
# Arguments:
# expname - name of experiment file in the experiments dir
# test_data.npz - file containing test data (i.e. generated from genTestData.py)
# nsample - # of evaluations at each location


import sys
import os
import time
import tensorflow as tf
import numpy as np
import utils as ut
from importlib import import_module
from config import *


def eval_model(exp, nsample):
    p = import_module('experiments.' + exp)
    # get data from DATA_PATH/mapname_test.npz
    mapname = os.path.basename(p.MAPFILE).split('.')[0]
    datafile = os.path.join(DATA_PATH, '{}_test.npz'.format(mapname))
    test_data = np.load(datafile)

    def mprint(s):
        sys.stdout.write(s+"\n")
        sys.stdout.flush()

    NTX = test_data['dist'].shape[1]

    # Check for saved weights
    saved = ut.ckpter(os.path.join(WTS_PATH, exp, 'iter_*.model.npz'))

    # Set up batches from test set
    batcher = ut.batcher([test_data['xy'], test_data['dist'], test_data['nint']], p.BSZ, shuffle=False)

    # Create placeholders for input
    dists = tf.placeholder(shape=(p.BSZ,NTX),dtype=tf.float32)
    nints = tf.placeholder(shape=(p.BSZ,NTX),dtype=tf.float32)
    itr = tf.placeholder(shape=(),dtype=tf.int32)

    # Load model
    net = p.md.model(NTX,dists,nints,itr)

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


    n = test_data['xy'].shape[0]
    preds = np.zeros((n, nsample, 2))
    power = np.zeros((n, nsample, net.NCHAN))

    # test each location "nsample" times
    n_batches = int(np.ceil((nsample * n) / float(p.BSZ)))

    # Eval Loop
    mprint(stamp()+"Starting model evaluation:")
    c = 0
    for i in xrange(n_batches):

        # Run training step & get next batch
        bdata = batcher.get_batch()
        fdict = {
            dists: bdata[1],
            nints: bdata[2],
            itr: -1,
        }
        out = sess.run(net.out, feed_dict=fdict)
        if batcher.pos < p.BSZ:
            size = p.BSZ - batcher.pos
            preds[-size:, c] = out[:size]
            c += 1
            if c < nsample:
                preds[:batcher.pos, c] = out[size:]
        else:
            preds[batcher.pos-p.BSZ : batcher.pos, c] = out

        if i % p.DISP_FREQ == 0:
            mprint(stamp() + "[%09d] batches evaluated of %d." % (i, n_batches))

    # Compute mean and max errror and number of beacons placed.
    dists = np.sum((preds - test_data['xy'][:,np.newaxis,:]) ** 2, axis=2)
    mean_err = np.sqrt(np.mean(dists))

    # Compute max error over batches of 20 test examples at the same location
    size = min(20, nsample)
    max_err = 0
    for i in range(nsample // size):
    	max_err += np.mean(np.max(dists[:,size*i:size*(i+1)], axis=-1))
    max_err = np.sqrt(max_err / (nsample // size))

    tx = np.load(saved.latest)
    t = np.argmax(tx['sensor'].reshape([625,-1]),axis=1)
    num_beacons = np.sum(t != 0)

    print("Mean Error: {}".format(mean_err))
    print("Max Error: {}".format(max_err))
    print("Beacons Placed: {}".format(num_beacons))


    # Save results
    results_dir = os.path.join(RESULTS_PATH, expname)
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    outfile = os.path.join(results_dir, 'results.npz')
    results = {'xy': test_data['xy'], 'preds': preds,
               'mean': mean_err, 'max': max_err, 'num_beacons': num_beacons}
    np.savez(outfile, **results)


if __name__== '__main__':
    if len(sys.argv) < 2:
        sys.exit('USAGE: eval_model.py expname [nsample]')

    expname = sys.argv[1]
    if len(sys.argv) < 3:
        nsample = 100
    else:
        nsample = int(sys.argv[2])
    eval_model(expname, nsample)
