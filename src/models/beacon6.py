"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 6: fixed beacons in rows of one channel creating a grid

import tensorflow as tf
import numpy as np

# Use with 8 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,9),dtype=np.float32)
    v[:,:,0] = 0.5

    v[5,2:23:5,1] = 1.0
    v[10,2:23:5,2] = 1.0
    v[15,2:23:5,3] = 1.0
    v[20,2:23:5,4] = 1.0
    v[2:23:5,5,5] = 1.0
    v[2:23:5,10,6] = 1.0
    v[2:23:5,15,7] = 1.0
    v[2:23:5,20,8] = 1.0

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
