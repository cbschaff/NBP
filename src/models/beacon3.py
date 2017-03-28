"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 3

import tensorflow as tf
import numpy as np

# Use with 16 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,17),dtype=np.float32)
    v[:,:,0] = 0.5

    v[(4-wn):(4+wn),(4-wn):(4+wn),1] = 1.0
    v[(4-wn):(4+wn),(8-wn):(8+wn),2] = 1.0
    v[(4-wn):(4+wn),(12-wn):(12+wn),3] = 1.0
    v[(4-wn):(4+wn),(16-wn):(16+wn),4] = 1.0

    v[(8-wn):(8+wn),(4-wn):(4+wn),5] = 1.0
    v[(8-wn):(8+wn),(8-wn):(8+wn),6] = 1.0
    v[(8-wn):(8+wn),(12-wn):(12+wn),7] = 1.0
    v[(8-wn):(8+wn),(16-wn):(16+wn),8] = 1.0

    v[(12-wn):(12+wn),(4-wn):(4+wn),9] = 1.0
    v[(12-wn):(12+wn),(8-wn):(8+wn),10] = 1.0
    v[(12-wn):(12+wn),(12-wn):(12+wn),11] = 1.0
    v[(12-wn):(12+wn),(16-wn):(16+wn),12] = 1.0

    v[(16-wn):(16+wn),(4-wn):(4+wn),13] = 1.0
    v[(16-wn):(16+wn),(8-wn):(8+wn),14] = 1.0
    v[(16-wn):(16+wn),(12-wn):(12+wn),15] = 1.0
    v[(16-wn):(16+wn),(16-wn):(16+wn),16] = 1.0

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
