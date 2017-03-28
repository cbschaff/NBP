"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 5: fixed beacons of 8 channels in alternating clusters (of different subsets of 16)
# like 1 2   5 6   1 2 ...
#      3 4   7 8   3 4 ...

#      5 6   1 2   5 6 ...
#      7 8   3 4   7 8 ...

import tensorflow as tf
import numpy as np

# Use with 16 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,17),dtype=np.float32)
    v[:,:,0] = 0.5

    y = [[  6,  10,   0,   4,],
     [  7,   9,   3,  13,],
     [  8,   6,   4,  15,],
     [  5,   1,  14,   0,],
     [  4,   7,   6,  11,],
     [ 15,   9,   5,   8,],
     [ 14,   9,   4,   8,],
     [  1,   6,  13,   7,],
     [  3,   8,   4,   6,],
     [ 12,   2,   9,  11,],
     [  0,   8,   1,   3,],
     [ 10,   2,   5,  14,],
     [  4,  11,   3,   0,],
     [  6,   5,  10,   1,],
     [ 14,   2,  11,  12,],
     [  6,   7,   9,  10,],
     [  9,  13,   0,  15,],
     [  3,  12,   10,  1,],
     [ 13,   0,   2,   4,],
     [ 14,   7,  12,   1,],
     [  5,   6,   2,  11,],
     [  1,   8,   4,   0,],
     [ 15,  13,  11,   7,],
     [  9,   2,   5,   4,],
     [  1,   8,  12,   7,],
     [  2,   14,  3,  15,],
     [ 13,   3,   7,   1,],
     [  4,   8,  11,   5,],
     [ 10,   6,   9,   7,],
     [  1,   0,  14,   9,],
     [  4,   6,  15,  12,],
     [  9,  15,   8,   0,],
     [  8,  10,  12,  14,],
     [  8,   4,   1,   7,],
     [  9,   3,   6,  12,],
     [ 11,   0,   5,   8,]]

    x= np.array(y)

    for j in xrange(6):
        for i in xrange(6):
            v[2+4*i,2+4*j,x[6*i+j,0]] = 1.0
            v[3+4*i,2+4*j,x[6*i+j,1]] = 1.0
            v[2+4*i,3+4*j,x[6*i+j,2]] = 1.0
            v[3+4*i,3+4*j,x[6*i+j,3]] = 1.0

    v = np.reshape(v,(17*625))


    lgsen = tf.Variable(v,trainable=False)
    self.weights['sensor'] = lgsen
    self.entropy = tf.constant(0)

    lgsen = tf.reshape(lgsen,[1,self.NTX,NCHAN+1,1])
    lgout = tf.to_float(tf.equal(lgsen,tf.nn.max_pool(lgsen,\
                            [1,1,self.NCHAN+1,1],[1,1,self.NCHAN+1,1],'VALID')))
    lgout = tf.reshape(lgout,[1,1,self.NTX,NCHAN+1])
    lgout = tf.slice(lgout,begin=[0,0,0,1],size=[-1,-1,-1,-1])
    return lgout
