"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 4: fixed beacons of a single channel on each wall

import tensorflow as tf
import numpy as np

# Use with 4 channels

def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,5),dtype=np.float32)
    v[:,:,0] = 0.5

    v[0:23,0,1] = 1.0
    v[1:24,24,2] = 1.0
    v[0,1:24,3] = 1.0
    v[24,0:23,4] = 1.0

    v = np.reshape(v,(5*625))

    lgsen = tf.Variable(v,trainable=False)
    self.weights['sensor'] = lgsen
    self.entropy = tf.constant(0)

    lgsen = tf.reshape(lgsen,[1,self.NTX,NCHAN+1,1])
    lgout = tf.to_float(tf.equal(lgsen,tf.nn.max_pool(lgsen,\
                            [1,1,self.NCHAN+1,1],[1,1,self.NCHAN+1,1],'VALID')))
    lgout = tf.reshape(lgout,[1,1,self.NTX,NCHAN+1])
    lgout = tf.slice(lgout,begin=[0,0,0,1],size=[-1,-1,-1,-1])
    return lgout
