# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:40:18 2016

@author: anguelos
"""
import numpy as np
import random
import lasagne
import sys
import theano.tensor as T
import theano
import time
import cPickle

from core import getLayerDict

def iterateMinibatches(inputs, targets, batchsize, **kwargs):
    def sortSamplesByClass(targets):
        if len(targets.shape)>1:
            clPos=targets.argmax(axis=1)
        else:
            clPos=targets.copy()
        perClassIdx={}
        perNonClassIdx={}
        for cNum in np.unique(clPos):
            perClassIdx[cNum]=(clPos==cNum).nonzero()[0]
            np.random.shuffle(perClassIdx[cNum])
            perNonClassIdx[cNum]=(clPos!=cNum).nonzero()[0]
            np.random.shuffle(perNonClassIdx[cNum])
        return (perClassIdx,perNonClassIdx)
    def createPairs(perClassIdx):
        newSz=sum([len(perClassIdx[k])*2 for k in perClassIdx.keys()])
        res=np.empty(newSz,dtype='int32')
        idx=0
        for k in perClassIdx.keys():
            l1=list(perClassIdx[k])
            l2=sorted(l1, key=lambda k: random.random())
            flattened=sum([list(l) for l in zip(l1,l2)],[])
            res[idx:idx+len(flattened)]=flattened
            idx+=len(flattened)
        newIdx1=sorted(range(0,len(res),2), key=lambda k: random.random())
        newIdx=sum([[i,i+1] for i in newIdx1],[])
        return res[newIdx]
    def createTriplets(perClassIdx,perNonClassIdx):
        newSz=sum([len(perClassIdx[k])*3 for k in perClassIdx.keys()])
        res=np.empty(newSz,dtype='int32')
        idx=0
        for k in perClassIdx.keys():
            l1=list(perClassIdx[k])
            l2=sorted(l1, key=lambda k: random.random())
            l3=perNonClassIdx[k][:len(l1)]
            flattened=sum([list(l) for l in zip(l1,l2,l3)],[])
            res[idx:idx+len(flattened)]=flattened
            idx+=len(flattened)
        newIdx1=sorted(range(0,len(res),3), key=lambda k: random.random())
        newIdx=sum([[i,i+1,i+2] for i in newIdx1],[])
        return res[newIdx]
    p={'shuffle':True,'batchMode':'normal'}
    p.update(kwargs)
    assert len(inputs) == len(targets)
    if p['batchMode']=='normal':
        indices = np.arange(len(inputs))
        if p['shuffle']:
            np.random.shuffle(indices)
    elif p['batchMode']=='pairs':
        print 'TARGET SHAPE:',targets.shape
        indices=createPairs(sortSamplesByClass(targets)[0])
    elif p['batchMode']=='triplets':
        indices=createTriplets(*sortSamplesByClass(targets))
    else:
        raise Exception('iterateMinibatches: mode should be one of ["normal","pairs","triplets"]')
    for start_idx in range(0, len(indices) - batchsize + 1, batchsize):
        excerpt = indices[start_idx:start_idx + batchsize]
        yield inputs[excerpt], targets[excerpt]


def train(model,trainSet,validationsSet,**kwargs):
    p={'loss':lasagne.objectives.categorical_crossentropy,'epochs':1,'momentum':.9,'minibatchSz':32,'reportMinibatch':False,'resumeFname':'','v':3,'outStream':sys.stderr,'lr':0.01,'lrMult':None,'regL2':0.0,'regL1':0.0,'minibatchesPerEpoch':-1}
    p.update(kwargs)
    input_var = getLayerDict(model)['input'].input_var
    target_var = T.ivector('targets')
    prediction = lasagne.layers.get_output(model)
    loss =p['loss'](prediction, target_var)
    if p['regL1']:
        d=getLayerDict(model)
        layers=dict([(d[k],p['regL1']) for k in range(d['sz'])])
        l1_penalty = lasagne.regularization.regularize_layer_params_weighted(layers, lasagne.regularization.l1)
        loss= loss+l1_penalty
    if p['regL2']:
        d=getLayerDict(model)
        layers=dict([(d[k],p['regL2']) for k in range(d['sz'])])
        l2_penalty = lasagne.regularization.regularize_layer_params_weighted(layers, lasagne.regularization.l1)
        loss= loss+l2_penalty
    loss = loss.mean()
    params = lasagne.layers.get_all_params(model, trainable=True)
    if p['lrMult']!=None:
        if type(p['lrMult'])==type([]):
            p['lrMult']=dict(enumerate(p['lrMult']))
        grads = theano.grad(loss, params)
        for idx, param in enumerate(params):
            if idx in p['lrMult'].keys():
                grads[idx] *= (p['lrMult'][idx])
        updates = lasagne.updates.nesterov_momentum(grads, params,learning_rate=p['lr'],momentum= p['momentum'])
    else:
        updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=p['lr'],momentum= p['momentum'])
    test_prediction = lasagne.layers.get_output(model, deterministic=True)
    test_loss = p['loss'](test_prediction,target_var)
    test_loss = test_loss.mean()
    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
    train_fn = theano.function([input_var, target_var], loss, updates=updates,allow_input_downcast=True)
    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],allow_input_downcast=True)
    e=p['outStream']
    for epoch in range(p['epochs']):
        train_err = 0
        train_batches = 0
        start_time = time.time()
        if p['v']>1:
            e.write('Epoch '+str(epoch)+': ')
        bc=0
        minibatchCount=0
        for batch in iterateMinibatches(trainSet['X'], trainSet['yLabs'], p['minibatchSz'], **p):
            if p['minibatchesPerEpoch']>0:
                if minibatchCount>p['minibatchesPerEpoch']:
                    break
                else:
                    minibatchCount+=1
            if p['v']>2:
                e.write('.')
            inputs, targets = batch
            train_err += train_fn(inputs, targets)
            if p['reportMinibatch']:
                verr, vacc = val_fn(inputs, targets)
                print 'E:%3d  B:%4d  Acc:%.3f  Err:%8.4f'%(epoch,bc,vacc,verr)
                bc+=1
            train_batches += 1
        # And a full pass over the validation data:
        val_err = 0
        val_acc = 0
        val_batches = 0
        if validationsSet:
            if p['v']>1:
                e.write('\nValidating: ')
            for batch in iterateMinibatches(validationsSet['X'],validationsSet['yLabs'], p['minibatchSz'],**p):
                if p['v']>2:
                    e.write('.')
                inputs, targets = batch
                err, acc = val_fn(inputs, targets)
                val_err += err
                val_acc += acc
                val_batches += 1
            if p['v']>2:
                e.write('\n')
        if p['v']>0:
            e.write("Epoch {} of {} took {:.3f}s\n".format(epoch + 1, p['epochs'], time.time() - start_time))
            e.write("  training loss:\t\t{:.6f}\n".format(train_err / train_batches))
            if validationsSet:
                e.write("  validation loss:\t\t{:.6f}\n".format(val_err / val_batches))
                e.write("  validation accuracy:\t\t{:.2f} %\n".format(val_acc / val_batches * 100))
        if p['v']>3 and len(p['resumeFname'])>0:
            cPickle.dump({'model':model,'epoch':epoch,'params':p},open(p['resumeFname'],'w'))
    return model
