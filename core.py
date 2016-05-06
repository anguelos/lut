# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:24:05 2016

@author: anguelos
"""
import theano
import lasagne
import numpy as np
import sys
#import theano.tensor as T

#from train import *


def getLayerDict(model):
    m=model
    res={}
    idx=-1
    while 'input_layer' in m.__dict__.keys() or 'input_layers' in m.__dict__.keys() :
        res[idx]=m
        idx-=1
        if type(m.name)==type(''):
            res[m.name]=m
        if 'input_layers' in m.__dict__.keys():
            m=m.input_layers[0]#inverse layer
        else:
            m=m.input_layer
    res['input']=m
    offset=(-idx)-1
    for k in range(-1,idx,-1):
        res[k+offset]=res[k]
    res['sz']=(-idx)-1
    return res

def getActivationsAsFuction(model,layer=-1,deterministic=True):
    d=getLayerDict(model)
    if layer<-d['sz'] or layer>=d['sz']:
        raise Exception('getActivationsAsFuction bad layer num '+str(layer)+' for a net with '+str(d['sz']))
    inputVar=d['input'].input_var
    return theano.function([inputVar],lasagne.layers.get_output(d[layer], deterministic=deterministic),allow_input_downcast=True)


def getActivations(model,data,layer=-1,deterministic=True):
    f=getActivationsAsFuction(model,layer,deterministic)
    return f(data)

def getActivationsAsDs(ds,network,layer=-1,**kwargs):
    p={'reshapeTo':None,'minibatchSz':64,'verbose':True}
    p.update(kwargs)
    network=getLayerDict(network)[layer]
    res=[]
    f=getActivationsAsFuction(network,layer)
    for dNum in range(len(ds)):
        if p['verbose']:
            print '.',
        d={'y':ds[dNum]['y'].copy(),'yLabs':ds[dNum]['yLabs'].copy()}
        nbSamples=len(ds[dNum]['yLabs'])
        resX=np.empty((nbSamples,)+network.output_shape[1:])
        ranges=[(k,k+p['minibatchSz']) for k in range(0,nbSamples,p['minibatchSz'])]
        ranges.append(((nbSamples/p['minibatchSz'])*p['minibatchSz'],nbSamples))
        if p['verbose']:
            print 'Processing in %d batches:'%len(ranges),
        for fromIdx,toIdx in ranges:
            if p['verbose']:
                print '.',
            resX[fromIdx:toIdx]=f(ds[dNum]['X'][fromIdx:toIdx])
        if p['verbose']:
            print 'done'
        if p['reshapeTo']!=None:
            resX=resX.reshape(p['reshapeTo'])
        d['X']=resX
        res.append(d)
    return res
    
    
def cloneColumn(model,**kwargs):
    p={'inputLayer':None,'weights':'shared','depth':-1}
    p.update(kwargs)
    def cloneDropoutTheano(doLayer,inpLayer,**kwargs):
        p={'namePrefix':'clone_'};p.update(kwargs)
        return lasagne.layers.DropoutLayer(inpLayer,p=doLayer.p,name=p['namePrefix']+str(doLayer))
    def cloneConv2DTheano(cLayer,inpLayer,**kwargs):
        p={'namePrefix':'clone_'};p.update(kwargs)
        c=cLayer
        if p['weights']=='shared':
            return lasagne.layers.Conv2DLayer(inpLayer,num_filters=c.num_filters,filter_size=c.filter_size,W=c.W,b=c.b,stride=c.stride,pad=c.pad,nonlinearity=c.nonlinearity,flip_filters=c.flip_filters,name=p['namePrefix']+str(cLayer.name))
        elif p['weights']=='copy':
            return lasagne.layers.Conv2DLayer(inpLayer,num_filters=c.num_filters,filter_size=c.filter_size,W=c.W.get_value(),b=c.b.get_value(),stride=c.stride,pad=c.pad,nonlinearity=c.nonlinearity,flip_filters=c.flip_filters,name=p['namePrefix']+str(cLayer.name))
        elif p['weights']=='random':
            return lasagne.layers.Conv2DLayer(inpLayer,num_filters=c.num_filters,filter_size=c.filter_size,stride=c.stride,pad=c.pad,nonlinearity=c.nonlinearity,flip_filters=c.flip_filters,name=p['namePrefix']+str(cLayer.name))
    def cloneConv2DDNNTheano(cLayer,inpLayer,**kwargs):
        p={'namePrefix':'clone_'};p.update(kwargs)
        c=cLayer
        if p['weights']=='shared':
            return lasagne.layers.dnn.Conv2DDNNLayer(inpLayer,num_filters=c.num_filters,filter_size=c.filter_size,W=c.W,b=c.b,stride=c.stride,pad=c.pad,nonlinearity=c.nonlinearity,flip_filters=c.flip_filters,name=p['namePrefix']+str(cLayer.name))
        elif p['weights']=='copy':
            return lasagne.layers.dnn.Conv2DDNNLayer(inpLayer,num_filters=c.num_filters,filter_size=c.filter_size,W=c.W.get_value(),b=c.b.get_value(),stride=c.stride,pad=c.pad,nonlinearity=c.nonlinearity,flip_filters=c.flip_filters,name=p['namePrefix']+str(cLayer.name))
        elif p['weights']=='random':
            return lasagne.layers.dnn.Conv2DDNNLayer(inpLayer,num_filters=c.num_filters,filter_size=c.filter_size,stride=c.stride,pad=c.pad,nonlinearity=c.nonlinearity,flip_filters=c.flip_filters,name=p['namePrefix']+str(cLayer.name))
    def clonePool2DTheano(cLayer,inpLayer,**kwargs):
        p={'namePrefix':'clone_'};p.update(kwargs)
        c=cLayer
        return lasagne.layers.Pool2DLayer(inpLayer,pool_size=c.pool_size,stride=c.stride,pad=c.pad,mode=c.mode,name=p['namePrefix']+str(cLayer.name))
    def cloneMaxPool2DLayer(cLayer,inpLayer,**kwargs):
        p={'namePrefix':'clone_'};p.update(kwargs)
        c=cLayer
        return lasagne.layers.MaxPool2DLayer(inpLayer,pool_size=c.pool_size,stride=c.stride,pad=c.pad,ignore_border=c.ignore_border,name=p['namePrefix']+str(cLayer.name))
    def cloneDenseTheano(cLayer,inpLayer,**kwargs):
        p={'namePrefix':'clone_'};p.update(kwargs)
        c=cLayer
        if p['weights']=='shared':
            return lasagne.layers.DenseLayer(inpLayer,num_units=c.num_units,nonlinearity=c.nonlinearity,W=c.W,b=c.b,name=p['namePrefix']+str(cLayer.name))
        elif p['weights']=='copy':
            return lasagne.layers.DenseLayer(inpLayer,num_units=c.num_units,nonlinearity=c.nonlinearity,W=c.W.get_value(),b=c.b.get_value(),name=p['namePrefix']+str(cLayer.name))
        elif p['weights']=='random':
            return lasagne.layers.DenseLayer(inpLayer,num_units=c.num_units,nonlinearity=c.nonlinearity,name=p['namePrefix']+str(cLayer.name))
        else:
            raise Exception('weights must be either "shared","copy", or "random"')
    def cloneSingleLayer(layer,inpLayer,**kwargs):
        if type(layer)==lasagne.layers.DenseLayer:
            return cloneDenseTheano(layer,inpLayer,**kwargs)
        elif type(layer)==lasagne.layers.Pool2DLayer:
            return clonePool2DTheano(layer,inpLayer,**kwargs)
        elif type(layer)==lasagne.layers.MaxPool2DLayer:
            return cloneMaxPool2DLayer(layer,inpLayer,**kwargs)
        elif type(layer)==lasagne.layers.Conv2DLayer:
            return cloneConv2DTheano(layer,inpLayer,**kwargs)
        elif type(layer)==lasagne.layers.dnn.Conv2DDNNLayer:
            return cloneConv2DDNNTheano(layer,inpLayer,**kwargs)
        elif type(layer)==lasagne.layers.DropoutLayer:
            return cloneDropoutTheano(layer,inpLayer,**kwargs)
        else:
            sys.stderr.write(str(type(layer))+'\n\n')
            raise NotImplementedError()
    d=getLayerDict(model)
    if p['depth']==-1:
        p['depth']=d['sz']
    if p['inputLayer']==None:
        inputLayer=lasagne.layers.InputLayer((d[-p['depth']].output_shape))
    else:
        inputLayer=p['inputLayer']
    network=inputLayer
    for lIdx in sorted(range(-1,-(1+p['depth']),-1)):
        network=cloneSingleLayer(d[lIdx],network,**p)
    return network


#def test(model,testSet,**kwargs):
#    p={'minibatchSz':32,'resume':'','v':3,'outStream':sys.stderr}
#    p.update(kwargs)
#    input_var = getLayerDict(model)['input']
#    target_var = T.ivector('targets')
#    prediction = lasagne.layers.get_output(model)
#    loss = lasagne.objectives.categorical_crossentropy(prediction, target_var)
#    loss = loss.mean()
#    #params = lasagne.layers.get_all_params(model, trainable=True)
#    #updates = lasagne.updates.nesterov_momentum(loss, params, learning_rate=p['lr'], momentum=p['momentum'])
#    test_prediction = lasagne.layers.get_output(model, deterministic=True)
#    test_loss = lasagne.objectives.categorical_crossentropy(test_prediction,target_var)
#    test_loss = test_loss.mean()
#    test_acc = T.mean(T.eq(T.argmax(test_prediction, axis=1), target_var),dtype=theano.config.floatX)
#    val_fn = theano.function([input_var, target_var], [test_loss, test_acc],allow_input_downcast=True)
#    e=p['outStream']
#    tst_err=0
#    tst_acc=0
#    tst_batches=0
#    for batch in iterateMinibatches(testSet['X'], testSet['yLabs'], p['minibatchSz'], shuffle=False):
#        if p['v']>2:
#            e.write('.')
#        inputs, targets = batch
#        err, acc = val_fn(inputs, targets)
#        tst_err += err
#        tst_acc += acc
#        tst_batches += 1
#    if p['v']>2:
#        e.write('\n')
#    if p['v']>0:
#        e.write("  testing loss:\t\t{:.6f}\n".format(tst_err / tst_batches))
#        e.write("  testing accuracy:\t\t{:.2f} %\n".format(tst_acc / tst_batches * 100))
#    return tst_acc / tst_batches * 100