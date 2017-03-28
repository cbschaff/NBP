"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Adds TensorFlow loss and optimization operations to a model

import tensorflow as tf
import numpy as np

def train(model,gt,lr,mom,wd=0., beacon_reg=0.):
    bsz = tf.to_int32(gt.get_shape()[0])
    n = tf.to_int32(model.out.get_shape()[0]) / bsz
    out = tf.reshape(model.out, [n, bsz, -1])
    max_err = tf.reduce_max(tf.reduce_sum((out - gt) ** 2, reduction_indices=2), reduction_indices=0)
    loss = tf.reduce_mean(max_err)
    obj = loss
    if wd > 0.:
        # Define L2 weight-decay on all non-bias vars
        reg = list()
        for k in model.weights.keys():
            wt = model.weights[k]
            if len(wt.get_shape()) > 1:
                reg.append(tf.nn.l2_loss(wt))

        reg = tf.add_n(reg)
        obj = obj + wd*reg
    if beacon_reg != 0.:
        sreg = tf.reduce_sum(model.sig) / model.NTX
        obj += beacon_reg * sreg

    if model.cost is not None:
        obj = obj + model.cost
    # Set up momentum trainer
    opt = tf.train.MomentumOptimizer(lr,mom)
    gv = opt.compute_gradients(obj)
    gv = [ [tf.clip_by_value(gvi[0],-1.0,1.0),gvi[1]] for gvi in gv]
    return loss, opt.apply_gradients(gv)
