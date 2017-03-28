"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 7: Fixed beacon clusters in most rooms around the exterior

import tensorflow as tf
import numpy as np

# Use with 8 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,9),dtype=np.float32)
    v[:,:,0] = 0.5


    v[(1-wn):(1+wn),(3-wn):(3+wn),1] = 1.0
    v[(1-wn):(1+wn),(6-wn):(6+wn),2] = 1.0
    v[(1-wn):(1+wn),(8-wn):(8+wn),3] = 1.0
    v[(1-wn):(1+wn),(11-wn):(11+wn),4] = 1.0
    v[(1-wn):(1+wn),(14-wn):(14+wn),5] = 1.0
    v[(1-wn):(1+wn),(16-wn):(16+wn),6] = 1.0
    v[(1-wn):(1+wn),(18-wn):(18+wn),7] = 1.0
    v[(1-wn):(1+wn),(21-wn):(21+wn),8] = 1.0
    v[(1-wn):(1+wn),(23-wn):(23+wn),1] = 1.0

    v[(5-wn):(5+wn),(21-wn):(21+wn),2] = 1.0
    v[(8-wn):(8+wn),(21-wn):(21+wn),3] = 1.0
    v[(11-wn):(11+wn),(21-wn):(21+wn),4] = 1.0
    v[(14-wn):(14+wn),(21-wn):(21+wn),5] = 1.0
    v[(17-wn):(17+wn),(21-wn):(21+wn),6] = 1.0

    v[(23-wn):(23+wn),(23-wn):(23+wn),7] = 1.0
    v[(23-wn):(23+wn),(20-wn):(20+wn),8] = 1.0
    v[(23-wn):(23+wn),(12-wn):(12+wn),1] = 1.0
    v[(23-wn):(23+wn),(8-wn):(8+wn),2] = 1.0
    v[(18-wn):(18+wn),(7-wn):(7+wn),3] = 1.0
    v[(23-wn):(23+wn),(5-wn):(5+wn),4] = 1.0
    v[(23-wn):(23+wn),(3-wn):(3+wn),5] = 1.0
    v[(23-wn):(23+wn),(1-wn):(1+wn),6] = 1.0

    v[(20-wn):(20+wn),(1-wn):(1+wn),7] = 1.0
    v[(17-wn):(17+wn),(1-wn):(1+wn),8] = 1.0
    v[(14-wn):(14+wn),(1-wn):(1+wn),2] = 1.0
    v[(11-wn):(11+wn),(1-wn):(1+wn),3] = 1.0
    v[(7-wn):(7+wn),(1-wn):(1+wn),4] = 1.0
    v[(3-wn):(3+wn),(1-wn):(1+wn),5] = 1.0

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
