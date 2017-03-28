"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 14: like 4, but with 8 channels and lines subdividing the room

import tensorflow as tf
import numpy as np

# Use with 8 channels

def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,9),dtype=np.float32)
    v[:,:,0] = 0.5

    v[0:24,0,1] = 1.0
    v[24,0:24,4] = 1.0
    v[1:25,24,2] = 1.0
    v[0,1:25,3] = 1.0

    v[8,1:24,5] = 1.0
    v[16,1:24,6] = 1.0
    v[1:24,8,7] = 1.0
    v[1:24,16,8] = 1.0

    v[8,8,5] = 0
    v[8,8,7] = 0
    v[8,16,5] = 0
    v[8,16,8] = 0
    v[16,16,6] = 0
    v[16,16,8] = 0
    v[16,8,6] = 0
    v[16,8,7] = 0

    v = np.reshape(v,(9*625))

    lgsen = tf.Variable(v,trainable=False)
    self.weights['sensor'] = lgsen
    self.entropy = tf.constant(0)

    lgsen = tf.reshape(lgsen,[1,self.NTX,NCHAN+1,1])
    lgout = tf.to_float(tf.equal(lgsen,tf.nn.max_pool(lgsen,\
                            [1,1,self.NCHAN+1,1],[1,1,self.NCHAN+1,1],'VALID')))
    lgout = tf.reshape(lgout,[1,1,self.NTX,NCHAN+1])
    lgout = tf.slice(lgout,begin=[0,0,0,1],size=[-1,-1,-1,-1])
    return lgout
