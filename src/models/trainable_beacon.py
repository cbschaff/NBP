"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Trainable Beacon

import tensorflow as tf
import numpy as np


# Beacon parameters

# Effective gamma is SFAC*(1 + (SDECAY*iter)^2)
# relative LR will be SFAC^2
SDECAY=np.sqrt(125.)/1e6
SFAC=1.
# Init std-dev of params
SINIT=1e-3

def beacon(self):
    NCHAN=self.NCHAN
    lgsen = tf.Variable(tf.random_normal([(NCHAN+1)*self.NTX],
                                             stddev=SINIT,dtype=tf.float32),
                            trainable=self.iter is not None)
    self.weights['sensor'] = lgsen

    lgsen = tf.reshape(lgsen,[1,self.NTX,NCHAN+1,1])
    lgout = tf.to_float(tf.equal(lgsen,
                tf.nn.max_pool(lgsen, [1,1,NCHAN+1,1], [1,1,NCHAN+1,1], 'VALID')))

    if self.iter is not None:
        fac=(SDECAY*tf.to_float(self.iter))
        fac = SFAC*(1. + fac*fac)

        lgsoft = tf.nn.softmax(tf.reshape(
            lgsen*fac,[-1,NCHAN+1]))
        lgsoft = tf.reshape(lgsoft,lgsen.get_shape())

        lgout = tf.cond(tf.logical_or(
                            tf.equal(self.iter,-1),
                            tf.greater(self.iter,900000)),
                        lambda: lgout, lambda: lgsoft)
        logp = tf.log(lgsoft)
        try: # tensorflow 1.0
            logp = tf.select(tf.is_nan(logp), tf.zeros_like(logp), logp)
        except:# tensorflow < 1.0
            logp = tf.where(tf.is_nan(logp), tf.zeros_like(logp), logp)
        self.entropy = -tf.reduce_sum(lgsoft * logp) / self.NTX
    else:
        self.entropy = tf.constant(0)


    lgout = tf.reshape(lgout,[1,1,self.NTX,NCHAN+1])
    lgout = tf.slice(lgout,begin=[0,0,0,1],size=[-1,-1,-1,-1])
    return lgout
