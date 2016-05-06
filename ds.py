# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:39:40 2016

@author: anguelos
"""

import numpy as np
import sys
import os
#from commands import getoutput as go
import sklearn.decomposition

datasetTmpDir='../.tmp/'


def reshapeDsInput(ds,newSize):
    res=[]
    for data in ds:
        newData=data.copy()
        newData['X']=newData['X'].reshape(newSize)
        res.append(newData)
    return res

def shuffleDs(ds):
    idx=np.arange(ds['X'].shape[0],dtype=int)
    np.random.shuffle(idx)
    res={}
    for k in ds.keys():
        res[k]=ds[k][idx]
    return res


def splitDs(ds,splitPcnt=.8):
    idx=np.arange(ds['X'].shape[0],dtype=int)
    part1Idx=idx[int(len(idx)*splitPcnt):]
    part2Idx=idx[:int(len(idx)*splitPcnt)]
    part1={}
    part2={}
    for k in ds.keys():
        part1[k]=ds[k][part1Idx]
        part2[k]=ds[k][part2Idx]
    return part1,part2


def sortDsByClass(ds,sortBy=None):
    if not(sortBy):
        sortBy=ds['yLabs']
    idx = np.arange(len(sortBy),dtype='int32')
    idx=idx[np.argsort(sortBy)]
    res={}
    for k in ds.keys():
        res[k]=ds[k][idx]
    return res


#def getSimpleWhitener(sampleStack):
#    sMean=sampleStack.mean(axis=0)
#    sStd=sampleStack.std(axis=0)+.000000001
#    print sMean.shape
#    def whitener(imgStack):
#        whitener.m=sMean
#        whitener.s=sStd
#        res=(imgStack-whitener.m)/whitener.s
#        return res
#    whitener(sampleStack)
#    return whitener
#
#
#def getZcaWhitener(sampleStack,mode='ZCA',epsilon=0.1):
#    raise NotImplementedError('Needs debugging')
#    sampleVectors=sampleStack.reshape([sampleStack.shape[0],-1])
#    sigma = np.dot(sampleVectors, sampleVectors.T)/sampleVectors.shape[1]
#    U,S,V = np.linalg.svd(sigma)
#    zcaMatrix=np.dot(np.dot(U, np.diag(1.0/np.sqrt(np.diag(S) + epsilon))), U.T)
#    def zcaWhitener(imgStack):
#        zcaWhitener.zcaMat=zcaMatrix
#        vecs=imgStack.reshape([imgStack.shape[0],-1])
#        wVecs=np.dot(zcaWhitener.zcaMat,vecs)
#        return wVecs.reshape(imgStack.shape)
#    zcaWhitener(sampleStack[:2,:])
#    return zcaWhitener
#
#
#def getWhitenDs(ds,whitenerName='std'):
#    if whitenerName=='zca':
#        w=getZcaWhitener(ds[0]['X'])
#    elif whitenerName=='std':
#        w=getSimpleWhitener(ds[0]['X'])
#    else:
#        raise Exception('Expecting whitener name ["zca","std"]')
#    res=[]
#    for d in ds:
#        resD=d.copy()
#        resD.update({'y':d['y'].copy(),'yLabs':d['yLabs'].copy(),'X':w(d['X'])})
#        res.append(resD)
#    return res

def blockNormaliseFeatureDs(dictList,**kwargs):
    p={'blockSz':256,'supress':[0]}
    p.update(kwargs)
    bSz=p['blockSz']
    res=[]
    for d in dictList:
        newD={}
        for k in d.keys():
            newD[k]=d[k].copy()
        newD['X']=np.empty(d['X'].shape)
        for k in range(0,d['X'].shape[1],bSz):
            newD['X'][:,k:k+bSz]=d['X'][:,k:k+bSz]
            newD['X'][:,[k+pos for pos in p['supress']]]=0
            newD['X'][:,k:k+bSz]/=newD['X'][:,k:k+bSz].sum(axis=1)[:,None]
        res.append(newD)
    return res


def standarizeFeatureDs(dictList):
    res=[]
    m=np.mean(dictList[0]['X'],axis=0)
    s=np.std(dictList[0]['X'],axis=0)
    for d in dictList:
        newD={}
        for k in d.keys():
            newD[k]=d[k].copy()
        newD['X']-=m
        newD['X']/=(s+.00000001)
        res.append(newD)
    return res


def hellingerFeatureDs(dictList,**kwargs):
    p={'pow':2.0}
    p.update(kwargs)
    res=[]
    for d in dictList:
        newD=dict(d)
        newD['X']=np.sign(d['X'])*np.abs(d['X'])**p['pow']
        res.append(newD)
    return res


def l2FeatureDs(dictList,**kwargs):
    p={'pow':2.0}
    p.update(kwargs)
    res=[]
    for d in dictList:
        newD=dict(d)
#resFeatures=self.features/((np.sum(self.features**2,1)**.5)+.00000000000001)[:,None]
        newD['X']=d['X']/((np.sum(d['X']**p['pow'],1)**(1.0/p['pow']))+.00000000000001)[:,None]
        res.append(newD)
    return res



def pcaFeatureDs(dictList,**kwargs):
    p={'pcaObj':None,'nbComponents':None}
    p.update(kwargs)
    res=[]
    if not(p['pcaObj']):
        pca=sklearn.decomposition.PCA(n_components=p['nbComponents'],copy=True).fit(dictList[0]['X'])
    else:
        pca=p['pcaObj']
    for inDs in dictList:
        res.append(dict(inDs))
        res[-1]['X']=pca.transform(inDs['X'])
    return res



def loadCsvFeatureFile(fname):
    lines=[l.split(',') for l in open(fname).read().split('\n') if len(l)]
    yLabs=[int(l[0].split('/')[-1].split('_')[0]) for l in lines]
    vals=[[int(c) for c in l[1:] if len(c)>0] for l in lines]
    yLabs=np.array(yLabs,dtype='int32')
    vals=np.array(vals,dtype='float')
    y=np.zeros([yLabs.shape[0],np.max(yLabs)+1],dtype=float)
    y[np.arange(yLabs.shape[0]),yLabs]=1
    return {'X':vals,'y':y,'yLabs':yLabs}


def loadMnist():
    if sys.version_info[0] == 2:
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    def download(filename,destDir, source='http://yann.lecun.com/exdb/mnist/'):
        print("Downloading %s" % filename)
        urlretrieve(source + filename, destDir+filename)
    import gzip
    def load_mnist_images(filename):
        if not os.path.exists(datasetTmpDir+filename):
            download(filename,datasetTmpDir)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=16)
        data = data.reshape(-1, 1, 28, 28)
        return data / np.float32(256)
    def load_mnist_labels(filename):
        if not os.path.exists(datasetTmpDir+filename):
            download(filename,datasetTmpDir)
        with gzip.open(filename, 'rb') as f:
            data = np.frombuffer(f.read(), np.uint8, offset=8)
        return data
    X_train = load_mnist_images('train-images-idx3-ubyte.gz')
    y_train = load_mnist_labels('train-labels-idx1-ubyte.gz')
    X_test = load_mnist_images('t10k-images-idx3-ubyte.gz')
    y_test = load_mnist_labels('t10k-labels-idx1-ubyte.gz')
    X_train, X_val = X_train[:-10000], X_train[-10000:]
    y_train, y_val = y_train[:-10000], y_train[-10000:]
    yTrainBv=np.zeros([y_train.shape[0],y_train.max()+1])
    yTrainBv[range(len(y_train)),y_train]=1
    yValBv=np.zeros([y_val.shape[0],y_val.max()+1])
    yValBv[range(len(y_val)),y_val]=1
    yTestBv=np.zeros([y_test.shape[0],y_test.max()+1])
    yTestBv[range(len(y_test)),y_test]=1
    return [{'X':X_train,'y':yTrainBv,'yLabs':y_train},{'X':X_val,'y':yValBv,'yLabs':y_val},{'X':X_test,'y':yTestBv,'yLabs':y_test}]


#LOAD SRS FEATURES
def srsFeatureLoader(trainFname,**kwargs):
    p={'shuffle':True,'testFname':None,'valFname':None,'valPart':0,'testPart':0,'outFname':None,'outDir':datasetTmpDir,'activeClass':0,'classExtraction':(lambda fn:[int(col) for col in fn.split('/')[-1].split('.')[0].split('_')])}
    p.update(kwargs)
    def loadFile(fname):
        lines=[l.split(',') for l in open(fname).read().split('\n') if len(l)>0]
        allLabels=np.array([p['classExtraction'](l[0])  for l in lines])
        labels=allLabels[:,p['activeClass']]
        features=np.array([[int(n) for n in l[1:]]  for l in lines])
        nbLabels=int(np.max(labels[:,1]))+1
        yVecs=np.zeros([labels.shape[0],nbLabels]);
        yVecs[np.arange(0,labels.shape[0],dtype='int32'),labels[:,1]]=1
        return {'X':features,'y':yVecs,'yLabs':labels,'auxLabs':allLabels}
    train= loadFile(trainFname)
    if p['shuffle']:
        train=shuffleDs(train)
    if p['valFname']:
        validation=loadFile(p['valFname'])
    elif p['valPart']>0:
        validation,train=splitDs(train,p['valPart'])
    else:
        validation=train
    if p['testFname']:
        test=loadFile(p['testFname'])
    elif p['testPart']>0:
        test,train=splitDs(train,p['testPart'])
    else:
        test=train
    if p['outFname]']:
        np.savez_compressed(open(p['outDir']+p['outFname'],'w'),train=train,val=validation,test=test)
    return [train,validation,test]


def loadMidlePatches(dsName,size=32):
    fnames=[(datasetTmpDir+'/%s_%d_train.npz'%(dsName,size)),'datasetTmpDir+/%s_%d_sample.npz'%(dsName,size),'../.tmp/%s_%d_test.npz'%(dsName,size)]
    res=[]
    for fname in fnames:
        f=np.load(fname)
        img=f['images'][:,2:3,:,:]
        res.append({'X':img,'y':f['labels'],'yLabs':np.argmax(f['labels'],axis=1)})
    return res


def loadAllPatches(dsName,size=32,patchesAsSamples=True):
    fnames=[(datasetTmpDir+'/%s_%d_train.npz'%(dsName,size)),datasetTmpDir+'/%s_%d_sample.npz'%(dsName,size),'../.tmp/%s_%d_test.npz'%(dsName,size)]
    res=[]
    for fname in fnames:
        f=np.load(fname)
        img=f['images'][:,:,:,:]
        if patchesAsSamples:
            y=np.empty([img.shape[0],img.shape[1],f['labels'].shape[1]])
            sIdx=np.empty([img.shape[0],img.shape[1]])
            for k in range(img.shape[1]):
                y[:,k,:]=f['labels']
                sIdx[:,k]=np.arange(img.shape[0])
            img=img.reshape([-1,1,size,size])
            y=y.reshape([-1,f['labels'].shape[1]])
            sIdx=sIdx.reshape([-1])
        else:
            y=f['labels']
            sIdx=np.arange(img.shape[0])
        res.append({'X':img,'y':y,'yLabs':np.argmax(y,axis=1),'sIdx':sIdx})
    return res

