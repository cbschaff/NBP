"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Propagation model - Thin Walls

import tensorflow as tf
import numpy as np

WFAC   = 0.1
DMIN   = 1.0

P0     = 6.25e-4
zeta   = 2.0
beta   = np.exp(-2 * WFAC)
RNZSTD = 0.01

def rxsignal(self,beacon,dist,nint):
    # Compute attenuated power
    txpow = tf.sqrt(P0 * (tf.maximum(dist,DMIN) ** -zeta) * (beta ** nint))

    # Sum received power with random phase and beacon probabilities
    txpow = tf.expand_dims(txpow,-1) * tf.reshape(beacon,
                                                  [1,self.NTX,self.NCHAN])

    txph = tf.random_uniform(txpow.get_shape(),0.0,2.0*np.pi)
    rxs = tf.reduce_sum(txpow*tf.sin(txph),[1])
    rxc = tf.reduce_sum(txpow*tf.cos(txph),[1])

    # Add noise
    rxs = rxs + tf.random_normal(rxs.get_shape(),stddev=RNZSTD)
    rxc = rxc + tf.random_normal(rxc.get_shape(),stddev=RNZSTD)

    self.rxpow = tf.minimum(tf.square(rxs) + tf.square(rxc),1.0)
    return self.rxpow
