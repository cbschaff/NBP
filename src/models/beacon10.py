"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 10: fixed beacons of 8 channels in alternating clusters (of different subsets of 16)

import tensorflow as tf
import numpy as np

# Use with 8 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,9),dtype=np.float32)
    v[:,:,0] = 0.5

    y = [[ 4,  3,  5,  1],
 [ 7,  2,  6,  8,],
 [ 3,  6,  5,  4,],
 [ 8,  4,  1,  7,],
 [ 6,  8,  5,  3,],
 [ 2,  4,  3,  7,],
 [ 1,  2,  5,  6,],
 [ 3,  8,  2,  7,],
 [ 8,  5,  6,  4,],
 [ 7,  1,  2,  3,],
 [ 8,  3,  6,  2,],
 [ 5,  4,  1,  7,],
 [ 3,  7,  8,  6,],
 [ 2,  1,  3,  4,],
 [ 8,  2,  7,  6,],
 [ 5,  1,  3,  4,],
 [ 8,  2,  7,  5,],
 [ 3,  4,  6,  1,],
 [ 1,  2,  6,  8,],
 [ 5,  7,  3,  4,],
 [ 7,  2,  8,  1,],
 [ 4,  5,  6,  3,],
 [ 8,  2,  3,  7,],
 [ 5,  1,  6,  4,],
 [ 2,  3,  8,  1,],
 [ 4,  7,  6,  5,],
 [ 8,  3,  1,  7,],
 [ 2,  6,  7,  8,],
 [ 1,  3,  5,  4,],
 [ 2,  7,  8,  6,],
 [ 8,  3,  2,  7,],
 [ 4,  6,  5,  1,],
 [ 8,  4,  3,  7,],
 [ 6,  2,  5,  1,],
 [ 4,  3,  7,  2,],
 [ 8,  5,  6,  1,]]

    x= np.array(y)

    for i in xrange(6):
        for j in xrange(6):
            v[2+4*i,2+4*j,x[6*i+j,0]] = 1.0
            v[3+4*i,2+4*j,x[6*i+j,1]] = 1.0
            v[2+4*i,3+4*j,x[6*i+j,2]] = 1.0
            v[3+4*i,3+4*j,x[6*i+j,3]] = 1.0

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
