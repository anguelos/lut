# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 20:16:40 2016

@author: anguelos
"""


import theano
import lasagne
import numpy as np

ZERO=0.0001

class RandomCropLayer(lasagne.layers.Layer):
    def __init__(self, incoming, cropSize, step=1, **kwargs):
        super(RandomCropLayer, self).__init__(incoming, **kwargs)
        if type(step)==type(1):
            step=(step,step)
        self.xStep=step[1]
        self.yStep=step[0]
        self.cropSize = tuple(cropSize)
        self._srng =theano.sandbox.rng_mrg.MRG_RandomStreams(lasagne.random.get_rng().randint(1, 2147462579))
        self.vGrid=(1+(incoming.output_shape[-2]-self.cropSize[0]))/self.yStep
        self.hGrid=(1+(incoming.output_shape[-1]-self.cropSize[1]))/self.xStep
        print self.vGrid,self.hGrid
        ridxX=self._srng.uniform((1,),0,self.hGrid+1,).floor().sum()
        self.didxX=theano.tensor.cast(ridxX, 'int64')*self.xStep
        ridxY=self._srng.uniform((1,),0,self.vGrid+1,).floor().sum()
        self.didxY=theano.tensor.cast(ridxY, 'int64')*self.yStep
    def get_output_shape_for(self, input_shape):
        return tuple(self.input_layer.output_shape[:-2]+self.cropSize)
    def get_output_for(self, input, **kwargs):
        return input[:,:,self.didxY:self.didxY+self.cropSize[0],self.didxX:self.didxX+self.cropSize[1]]


def __createLBPFilterWeights__(nbSamples,radius,fsz,**kwargs):
    defaultArgs={'zeroVal':.001,'filterSum':1.0}
    defaultArgs.update(kwargs)
    maxSupport=int((fsz-1)/2)
    res=np.random.rand(nbSamples+1,1,fsz,fsz)*defaultArgs['zeroVal']
    a=.00000001+2*np.pi*np.arange(nbSamples)/nbSamples
    x=np.cos(a)*radius+maxSupport;y=np.sin(a)*radius+maxSupport
    left=np.floor(x).astype('int')
    right=left+1
    top=np.floor(y).astype('int')
    bottom=top+1
    lCoef=(right-x)
    rCoef=1-lCoef
    tCoef=(bottom-y)
    bCoef=(1-tCoef)
    res[np.arange(nbSamples,dtype='int'),0,left,top]=lCoef*tCoef
    res[np.arange(nbSamples,dtype='int'),0,left,bottom]=lCoef*bCoef
    res[np.arange(nbSamples,dtype='int'),0,right,top]=rCoef*tCoef
    res[np.arange(nbSamples,dtype='int'),0,right,bottom]=rCoef*bCoef
    res[-1,0,maxSupport,maxSupport]=1
    for fNum in range(nbSamples+1):
        res[fNum,:,:,:]/=(res[fNum,:,:,:].sum()+.00000001)
        res[fNum,:,:,:]*=defaultArgs['filterSum']
    return res


def __createDeltaWeights__(nbSamples,**kwargs):
    defaultArgs={'zeroVal':.001,'filterSum':1.0}
    defaultArgs.update(kwargs)
    res=np.random.rand(nbSamples,nbSamples+1,1,1)*defaultArgs['zeroVal']
    res[:,:-1,0,0]=np.eye(nbSamples)
    res[:,-1,0,0]-=1
    return res


def __createUniformityWeights__(nbSamples,**kwargs):
    defaultArgs={'zero':ZERO,'filterSum':1.0}
    defaultArgs.update(kwargs)
    res=np.random.rand([2**nbSamples,nbSamples,1,1])*defaultArgs['zero']
    for p in range(2**nbSamples):
        pArray= [float(l=='1')*(1+nbSamples)-nbSamples for l in bin(p+2**nbSamples)[3:]]
        res[p,:,0,0]=pArray
    return res#res/nbSamples


def __createVocabularyWeights__(nbSamples):
    resW=np.zeros([2**nbSamples,nbSamples,1,1])
    resb=np.zeros(2**nbSamples)
    for p in range(2**nbSamples):
        #pArray= [float(l=='1')*(1+nbSamples)-nbSamples for l in bin(p+2**nbSamples)[3:]]
        #res[p,:,0,0]=pArray
        resW[p,:,0,0]=[float(l=='1')*(nbSamples+1)-nbSamples for l in bin(p+2**nbSamples)[3:]]
        #resb[p]=1-sum([float(l=='1') for l in bin(p+2**nbSamples)[3:]])
    resb[0]=1
    return (resW,resb)


def createLbpLayers(**kwargs):
    p={'nbSamples':8,'radius':3,'fsz':9,'inputSz':(64,64),'lbpWeightInit':True,'zeroVal':.001,'inputLayer':None}
    p.update(kwargs)
    inputSz=p['inputSz']
    #nbSamples=p['nbSamples']
    fsz=p['fsz']
    radius=p['radius']
    if p['inputLayer']==None:
        network = lasagne.layers.InputLayer(shape=(None, 1, inputSz[0], inputSz[1]))
    else:
        network = p['inputLayer']
    network = samplingFilters = lasagne.layers.Conv2DLayer(network,name=('lbp%d_%dsmpl'%(int(p['ndSamples']),int(p['radius']))), num_filters=p['nbSamples']+1, filter_size=(fsz, fsz),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='same')
    network = deltaFilters = lasagne.layers.Conv2DLayer(network,name=('lbp%d_%delta'%(int(p['ndSamples']),int(p['radius']))), num_filters=p['nbSamples'], filter_size=(1, 1),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='same')
    network = deltaToLabelFilters = lasagne.layers.Conv2DLayer(network,name=('lbp%d_%label'%(int(p['ndSamples']),int(p['radius']))), num_filters=2**p['nbSamples'], filter_size=(1, 1),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='same')
    #network = NormalisedMaxActivationLayer(network)
    #network = lasagne.layers.FeatureWTALayer(network,256,axis=1)
    #network = lasagne.layers.Pool2DLayer(network,(inputSz[0]/4,inputSz[1]),mode='average_exc_pad',pad=(4,0))
    network = lasagne.layers.Pool2DLayer(network,(inputSz[0]/4,inputSz[1]),name=('lbp%d_%dpool'%(int(p['ndSamples']),int(p['radius']))),mode='average_inc_pad')
    if p['lbpWeightInit']:
        samplingW,samplingb=samplingFilters.get_params()
        samplingW.set_value(__createLBPFilterWeights__(p['nbSamples'],radius,fsz,zeroVal=p['zeroVal']).astype('float32'))
        deltaW,deltab=deltaFilters.get_params()
        deltaW.set_value(__createDeltaWeights__(p['nbSamples'],zeroVal=p['zeroVal']).astype('float32'))
        labelW,labelb=deltaToLabelFilters.get_params()
        W,b=__createVocabularyWeights__(p['nbSamples'])
        labelW.set_value(W.astype('float32'))
        labelb.set_value(b.astype('float32'))
        return network
    return network

