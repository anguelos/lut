# -*- coding: utf-8 -*-
"""
Created on Mon Mar 21 10:42:21 2016

@author: anguelos
"""

from core import getLayerDict
#from core import cloneColumn
from train import iterateMinibatches
from view import getModelArchStr
#from loss import siamesePairLoss
#from loss import siameseTripletLoss
import lasagne
import theano.tensor as T
import theano
import sys
import time
import numpy as np
import scipy.spatial.distance

def getInverseEncoder(model,**kwargs):
    p={'stich':True,'inputVar':None}
    p.update(kwargs)
    d=getLayerDict(model)
    if p['stich']:
        network=model
    else:
        if p['inputVar']:
            network=lasagne.layers.InputLayer(None,p['inputVar'])
        else:
            network=lasagne.layers.InputLayer(model.output_shape)
    for k in range(len(d.keys())/2-2,-1,-1):
        if d[k].name:
            name='inv_'+d[k].name
        else:
            name='inv_l'+str(k)
        network=lasagne.layers.InverseLayer(network,d[k],name=name)
        print k,':\n',getModelArchStr(network)
    return network


def trainMetric(model,ds,**kwargs):
    p={'loss':siameseTripletLoss,'epochs':1,'momentum':.9,'minibatchSz':32,'reportMinibatch':False,'resumeFname':'','v':3,'outStream':sys.stderr,'lr':0.01,'lrMult':None,'regL2':0.0,'regL1':0.0}
    p.update(kwargs)
    source=ds['X']
    dest=ds['y']
    e=sys.stderr
    if p['loss']==siamesePairLoss:
        p['batchMode']='pairs'
        if p['minibatchSz']%2!=0:
            raise Exception('When training with a siamese pair loss you minibatch size must be a multiple of 2')
    elif p['loss']==siameseTripletLoss:
        p['batchMode']='triplets'
        if p['minibatchSz']%3!=0:
            raise Exception('When training with a metric learning triplet loss you minibatch size must be a multiple of 3')
    else:
        raise Exception('Loss must be either siamese pair or siamese triplte loss')
    input_var = getLayerDict(model)['input'].input_var
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(model)
    loss =p['loss'](prediction, target_var)
    params = lasagne.layers.get_all_params(model, trainable=True)
    grads = theano.grad(loss, params)
    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=p['lr'],momentum= p['momentum'])
    updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=p['lr'],momentum= p['momentum'])
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)
    for epoch in range(p['epochs']):
        train_err = 0
        train_batches = 0
        if p['v']>1:
            e.write('Epoch '+str(epoch)+': ')
        #bc=0
        for batch in iterateMinibatches(source, dest, p['minibatchSz'], **p):
            if p['v']>2:
                e.write('.')
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            train_batches += 1
#    batches=iterateMinibatches(source,dest,**p)


siamesePairLoss=None

def siameseTripletLoss(activations,labels):
    """
    based on Sumit Chopra et 2005
    
    """
    Q=1.0*activations.shape[1]
    #labels=T.argmax(gt,axis=1)
    x1=activations[0:-2,:]
    x2=activations[1:-1,:]
    y=T.eq(labels[0:-2],labels[1:-1])
    EW=T.sum(abs(x1-x2))
    loss=(1-y)*((2/Q)*(EW**2))+y*(2*Q*(np.e**(EW*-2.77/Q)))
    return loss


def getMAP(ds,**kwargs):
    p={'metric':'euclidean'}
    p.update(kwargs)
    dm=scipy.spatial.distance.cdist(ds['X'],ds['X'],metric=p['metric'])
    #print dm.shape
    pos=np.argsort(dm,axis=1).astype('int32')
    idxMat=(np.ones(pos.shape).cumsum(axis=0)-1).astype('int32')
    correctMat=ds['yLabs'][idxMat]==ds['yLabs'][pos]
    correctMat=correctMat[:,1:]
    #print correctMat.sum(axis=1)
    accuracy=correctMat[:,0].mean()
    precisionMat=correctMat.cumsum(axis=1)/np.ones(correctMat.shape).cumsum(axis=1)
    return ((precisionMat*correctMat).sum(axis=1)/correctMat.sum(axis=1)).mean(),accuracy
