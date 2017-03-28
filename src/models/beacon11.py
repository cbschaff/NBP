"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

# Beacon model 11: Fixed beacons clusters in most rooms

import tensorflow as tf
import numpy as np

# Use with 8 channels

wn=1
def beacon(self):
    NCHAN=self.NCHAN

    v = np.zeros((25,25,9),dtype=np.float32)
    v[:,:,0] = 0.5

    v[(1-wn):(1+wn),(2-wn):(2+wn),1] = 1.0
    v[(1-wn):(1+wn),(10-wn):(10+wn),2] = 1.0
    v[(1-wn):(1+wn),(12-wn):(12+wn),3] = 1.0
    v[(1-wn):(1+wn),(16-wn):(16+wn),4] = 1.0
    v[(1-wn):(1+wn),(23-wn):(23+wn),5] = 1.0

    v[(7-wn):(7+wn),(2-wn):(2+wn),6] = 1.0
    v[(11-wn):(11+wn),(2-wn):(2+wn),7] = 1.0
    v[(15-wn):(15+wn),(2-wn):(2+wn),8] = 1.0
    v[(17-wn):(17+wn),(2-wn):(2+wn),1] = 1.0
    v[(20-wn):(20+wn),(2-wn):(2+wn),2] = 1.0
    v[(24-wn):(24+wn),(2-wn):(2+wn),3] = 1.0

    v[(24-wn):(24+wn),(8-wn):(8+wn),4] = 1.0
    v[(21-wn):(21+wn),(7-wn):(7+wn),5] = 1.0
    v[(18-wn):(18+wn),(7-wn):(7+wn),6] = 1.0
    v[(16-wn):(16+wn),(7-wn):(7+wn),7] = 1.0
    v[(10-wn):(10+wn),(7-wn):(7+wn),5] = 1.0

    v[(18-wn):(18+wn),(9-wn):(9+wn),1] = 1.0
    v[(14-wn):(14+wn),(9-wn):(9+wn),2] = 1.0
    v[(12-wn):(12+wn),(9-wn):(9+wn),3] = 1.0
    v[(6-wn):(6+wn),(10-wn):(10+wn),4] = 1.0

    v[(21-wn):(21+wn),(14-wn):(14+wn),5] = 1.0
    v[(18-wn):(18+wn),(14-wn):(14+wn),6] = 1.0
    v[(16-wn):(16+wn),(13-wn):(13+wn),7] = 1.0
    v[(11-wn):(11+wn),(12-wn):(12+wn),8] = 1.0

    v[(15-wn):(15+wn),(17-wn):(17+wn),1] = 1.0
    v[(13-wn):(13+wn),(17-wn):(17+wn),2] = 1.0
    v[(9-wn):(9+wn),(16-wn):(16+wn),3] = 1.0
    v[(5-wn):(5+wn),(16-wn):(16+wn),4] = 1.0

    v[(21-wn):(21+wn),(21-wn):(21+wn),5] = 1.0
    v[(19-wn):(19+wn),(20-wn):(20+wn),6] = 1.0
    v[(17-wn):(17+wn),(21-wn):(21+wn),7] = 1.0
    v[(15-wn):(15+wn),(21-wn):(21+wn),8] = 1.0
    v[(13-wn):(13+wn),(21-wn):(21+wn),1] = 1.0
    v[(11-wn):(11+wn),(21-wn):(21+wn),2] = 1.0
    v[(9-wn):(9+wn),(21-wn):(21+wn),3] = 1.0

    v[(19-wn):(19+wn),(24-wn):(24+wn),4] = 1.0

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
