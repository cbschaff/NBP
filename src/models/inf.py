"""
Copyright (C) 2017  Charles Schaff, David Yunis, Ayan Chakrabarti,
Matthew R. Walter. See LICENSE.txt for details.
"""

import tensorflow as tf
import numpy as np

NCHAN=8 # No. of transmitter channels
smod = []
pmod = []

nUnits=1024
nGroup=3
nLayer=2

# Rate at which batch-norm population averages decay
_bndecay=0.99
_bneps=1e-4

class model:
    # Add a fully connected layer
    def fc(self,inp,nIn,nOut,name,ifrelu=True,ifbn=True):
        # xavier init
        sq = np.sqrt(3.0 / np.float32(nIn))
        w = tf.Variable(tf.random_uniform([nIn,nOut],minval=-sq,maxval=sq,dtype=tf.float32))
        self.weights[name+'_W'] = w

        # constant init
        b = tf.Variable(tf.constant(0,shape=[nOut],dtype=tf.float32))
        self.weights[name+'_b'] = b
        b = np.sqrt(2)*b # Learning rate x 2

        # Add FC layer
        out = tf.matmul(inp,w)

        # Batch-normalization
        if ifbn:
            b_shape = [1,nOut]

            # Add batch-norm vars
            try: # tensorflow 1.0
                mn = tf.Variable(tf.zeros_initializer(dtype=tf.float32)(shape = b_shape), trainable=False)
                vr = tf.Variable(tf.zeros_initializer(dtype=tf.float32)(shape = b_shape), trainable=False)
            except: # tensorflow < 1.0
                mn = tf.Variable(tf.zeros_initializer(
                    shape=b_shape,dtype=tf.float32),trainable=False)
                vr = tf.Variable(tf.zeros_initializer(
                    shape=b_shape,dtype=tf.float32),trainable=False)

            self.weights[name+'_bnm'] = mn
            self.weights[name+'_bnv'] = vr

            if self.iter is not None:
                out_m, out_v = tf.nn.moments(out,axes=[0])
                out_m = tf.reshape(out_m,b_shape)
                out_v = tf.reshape(out_v,b_shape)

                self.bnops.append(tf.assign(mn,mn*_bndecay + out_m*(1.-_bndecay)).op)
                self.bnops.append(tf.assign(vr,vr*_bndecay + out_v*(1.-_bndecay)).op)

                out_m = tf.cond(tf.equal(self.iter,-1), lambda: mn, lambda: out_m)
                out_v = tf.cond(tf.equal(self.iter,-1), lambda: vr, lambda: out_v)

                out = tf.nn.batch_normalization(out,out_m,out_v,None,None,_bneps)
            else:
                out = tf.nn.batch_normalization(out,mn,vr,None,None,_bneps)

        out = out + b
        # ReLU
        if ifrelu:
            out = tf.nn.relu(out)

        return out

    # Pool groups of units in a flat output
    def pool(self,inp,npool):
        inp = tf.expand_dims(tf.expand_dims(inp,0),-1)
        out = tf.nn.max_pool(inp,[1,1,npool,1],[1,1,npool,1],'VALID')
        return tf.squeeze(out,[0,-1])

    # Build network
    def __init__(self,NTX,dists,nints,iter):
        self.weights = {}
        self.bnops = []
        self.NTX=NTX
        self.NCHAN=NCHAN
        self.iter=iter

        self.sig = smod.beacon(self)
        sig = pmod.rxsignal(self,self.sig,dists,nints)


        out = sig
        prev = self.NCHAN
        for i in range(nGroup):
            for j in range(nLayer):
                out = self.fc(out,prev,nUnits,
                              'fc'+str(i+1)+'_'+str(j+1))
                prev=nUnits
            out = self.pool(out,4)
            prev= prev // 4

        # Final localizer output
        self.out = self.fc(out,prev,2,'pred',ifrelu=False,ifbn=False)
