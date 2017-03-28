"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 8: fixed beacons of a single channel on diagonals intersecting in the middle

import tensorflow as tf
import numpy as np

# Use with 4 channels

def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,5),dtype=np.float32)
    v[:,:,0] = 0.5

    for i in xrange(13):
        v[i,i,1] = 1.0
        v[24-i,24-i,2] = 1.0
        v[i,24-i,3] = 1.0
        v[24-i,i,4] = 1.0
    v[12,12,2:] = 0

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
