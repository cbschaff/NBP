"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 12: fixed beacons of 8 channels in alternating clusters, but less clusters than 10

import tensorflow as tf
import numpy as np

# Use with 8 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,9),dtype=np.float32)
    v[:,:,0] = 0.5

    y = [[ 6,  3,  7,  1],
 [ 2,  4,  5,  8],
 [ 1,  6,  2,  3],
 [ 4,  5,  2,  8],
 [ 7,  3,  1,  4],
 [ 5,  7,  6,  8],
 [ 6,  2,  4,  3],
 [ 8,  1,  5,  7],
 [ 3,  6,  7,  2],
 [ 7,  1,  5,  2],
 [ 3,  7,  8,  4],
 [ 1,  2,  6,  5],
 [ 6,  8,  4,  3],
 [ 5,  3,  7,  1],
 [ 5,  2,  6,  3],
 [ 6,  7,  1,  8]]

    x= np.array(y)

    for i in xrange(4):
        for j in xrange(4):
            v[2+6*i,2+6*j,x[4*i+j,0]] = 1.0
            v[3+6*i,2+6*j,x[4*i+j,1]] = 1.0
            v[2+6*i,3+6*j,x[4*i+j,2]] = 1.0
            v[3+6*i,3+6*j,x[4*i+j,3]] = 1.0

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
