# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 19:29:23 2016

@author: anguelos
"""
import lasagne
import sys
try:
    import lasagne.layers.dnn
except:
    sys.stderr.write('Failed to load lasagne.layers.dnn')
from core import getLayerDict
from layers import RandomCropLayer

def createLenet(**kwargs):
    """
    Creates a network similar to Lenet-5 but with ReLU and dropout
    
    Kwargs:
        inputLayer: a lasagne layer that provides a 4D tensor as it's activations, if None it is created. By default None
        inputSz (tuple): a tuple containing the width and height of the InputLayer if one is to be created. By default (28,28)
        outputSz (int): the number of output newrons, By default 10
    """
    p={'inputSz':(28,28),'inputVar':None,'outputSz':10,'inputLayer':None}
    p.update(kwargs)
    if p['inputLayer']==None:
        network = lasagne.layers.InputLayer(shape=(None, 1, p['inputSz'][0], p['inputSz'][1]),input_var=p['inputVar'])
    else:
        network= p['inputLayer']
    network = lasagne.layers.Conv2DLayer(network,name='conv1', num_filters=32, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform())
    network = lasagne.layers.MaxPool2DLayer(network,name='pool1', pool_size=(2, 2))
    network = lasagne.layers.Conv2DLayer(network,name='conv2', num_filters=32, filter_size=(5, 5),nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.MaxPool2DLayer(network,name='pool2', pool_size=(2, 2))
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),name='fc1',num_units=256,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),name='fc2',num_units=p['outputSz'],nonlinearity=lasagne.nonlinearities.softmax)
    return network


def createMlp(**kwargs):
    """
    Kwargs:
        activations: a lasagne nonlinearity, or list containing a lasagne non-linearity for every non-output layer of the MLP
        dropout: a float or a list of floats containing the dropout applied between each layer
        nbNeurons (tuple): a tuple containing the number of neurons per non-output layer
        outputSz (int): the number of output newrons, By default 10
    """
    p={'activations':lasagne.nonlinearities.rectify,
       'dropout':.5,
       'nbNeurons':(1024,512),
       'outActivation':lasagne.nonlinearities.softmax,
       'outSz':10,
       'inputLayer':None,
       'inputSz':(None,2048)}
    p.update(kwargs)
    try:
        p['activations']=list(p['activations'])
    except:
        p['activations']=list([p['activations']]*len(p['nbNeurons']))
    try:
        p['dropout']=list(p['dropout'])
    except:
        p['dropout']=list([p['dropout']]*len(p['nbNeurons']))
    if len(p['nbNeurons'])!=len(p['activations']) or len(p['nbNeurons'])!=len(p['dropout']):
        raise Exception('Architecture parameters dissagre')
    if p['inputLayer']==None:
        network=lasagne.layers.InputLayer(p['inputSz'])
    else:
        network=p['inputLayer']
    for k in range(len(p['nbNeurons'])):
        network=lasagne.layers.DenseLayer(network,num_units=p['nbNeurons'][k],nonlinearity=p['activations'][k],name=('fc%d'%k))
        if p['dropout'][k]>0:
            network=lasagne.layers.dropout(network,p=p['dropout'][k],name=('do%d'%k))
    network=lasagne.layers.DenseLayer(network,num_units=p['outSz'],nonlinearity=p['outActivation'],name=('fc%d'%len(p['nbNeurons'])))
    return network


def createLluNet(outSize=10,inSize=(40,40),step=8):
    network = lasagne.layers.InputLayer(shape=(None, 1, inSize[0], inSize[1]),name='input')
    network = RandomCropLayer(network,(32,32),8)
    network = lasagne.layers.Conv2DLayer(network,name='conv1', num_filters=96, filter_size=(5,5),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='valid')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network,name='pool1',pool_size=(3,3),stride=2,pad=0)
    network = lasagne.layers.Conv2DLayer(network,name='conv2', num_filters=256, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='valid')
    network = lasagne.layers.LocalResponseNormalization2DLayer(network)
    network = lasagne.layers.MaxPool2DLayer(network,name='pool2',pool_size=(3,3),stride=2,pad=1)
    network = lasagne.layers.Conv2DLayer(network,name='conv3', num_filters=384, filter_size=(3,3),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='valid')
    network = lasagne.layers.MaxPool2DLayer(network,name='pool3',pool_size=(3,3),stride=2,pad=1)
    network = lasagne.layers.Conv2DLayer(lasagne.layers.dropout(network, p=.5), num_filters=512, filter_size=(1,1),nonlinearity=lasagne.nonlinearities.rectify,W=lasagne.init.GlorotUniform(),pad='valid')
    network = lasagne.layers.DenseLayer(network,name='fc1',num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),name='fc2',num_units=1024,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.DenseLayer(lasagne.layers.dropout(network, p=.5),name='fc3',num_units=outSize,nonlinearity=lasagne.nonlinearities.softmax)
    return network


def transfuseCaffeWeights(lasagneNetDict,caffeNet):
    """
    Copies the weights from a pretrained caffe model in to a dictionary whith names and lasagne layer pairs.
    The names in the dictionary must match the names of the caffe model
    
    Args:
    lasagneNetDict (dictionary): lasagne layers of the network indexed by the names they have in the caffe model
    caffeNet (caffe.Net): A pretrained caffe model
    """
    layers_caffe = dict(zip(list(caffeNet._layer_names), caffeNet.layers))
    for name, layer in lasagneNetDict.items():
        try:
            print layer.W.get_value().shape,'->',layers_caffe[name].blobs[0].data.shape
            print layer.b.get_value().shape,'->',layers_caffe[name].blobs[1].data.shape
            sh=layer.W.get_value().shape
            if len(sh)==2 and (sh[1],sh[0])==layers_caffe[name].blobs[0].data.shape:
                layer.W.set_value(layers_caffe[name].blobs[0].data.T.copy())
            else:
                layer.W.set_value(layers_caffe[name].blobs[0].data.copy())
            layer.b.set_value(layers_caffe[name].blobs[1].data.copy())
            print 'Layer ',name,' OK!'
        except AttributeError:
            print 'Issue for layer ',name
            continue


def transfuseCaffeWeightsSimple(model,caffeNet):
    """
    Copies the weights from a pretrained caffe model in to a dictionary whith names and lasagne layer pairs.
    The names in the dictionary must match the names of the caffe model
    
    Args:
    lasagneNetDict (dictionary): lasagne layers of the network indexed by the names they have in the caffe model
    caffeNet (caffe.Net): A pretrained caffe model
    """
    layers_caffe = dict(zip(list(caffeNet._layer_names), caffeNet.layers))
    d=getLayerDict(model)
    lasagneLayerNames=[name for name in d.keys() if type(name)==str and name!='input']
    
    for name in lasagneLayerNames:
        layer=d[name]
        try:
            print layer.W.get_value().shape,'->',layers_caffe[name].blobs[0].data.shape
            print layer.b.get_value().shape,'->',layers_caffe[name].blobs[1].data.shape
            sh=layer.W.get_value().shape
            if len(sh)==2 and (sh[1],sh[0])==layers_caffe[name].blobs[0].data.shape:
                layer.W.set_value(layers_caffe[name].blobs[0].data.T.copy())
            else:
                layer.W.set_value(layers_caffe[name].blobs[0].data.copy())
            layer.b.set_value(layers_caffe[name].blobs[1].data.copy())
            print 'Layer ',name,' OK!'
        except AttributeError:
            print 'Issue for layer ',name
            continue



def __getNinCifar10ArchDict__(inputLayer=None):
    net = {}
    if inputLayer==None:
        net['input'] = lasagne.layers.InputLayer((None,3,224,224))
    else:
        net['input'] = inputLayer
#    net['input'] = lasagne.layers.InputLayer((None, 3, 32, 32))
#    net['conv1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['input'], num_filters=192, filter_size=5, pad=2, flip_filters=False)
#    net['cccp1'] = lasagne.layers.dnn.Conv2DDNNLayer(net['conv1'], num_filters=160, filter_size=1, flip_filters=False)
#    net['cccp2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['cccp1'], num_filters=96, filter_size=1, flip_filters=False)
#    net['pool1'] = lasagne.layers.Pool2DLayer(net['cccp2'], pool_size=3, stride=2, mode='max', ignore_border=False)
#    net['drop3'] = lasagne.layers.DropoutLayer(net['pool1'], p=0.5)
#    net['conv2'] = lasagne.layers.dnn.Conv2DDNNLayer(net['drop3'], num_filters=192, filter_size=5, pad=2, flip_filters=False)
#    net['cccp3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['conv2'], num_filters=192, filter_size=1, flip_filters=False)
#    net['cccp4'] = lasagne.layers.dnn.Conv2DDNNLayer(net['cccp3'], num_filters=192, filter_size=1, flip_filters=False)
#    net['pool2'] = lasagne.layers.Pool2DLayer(net['cccp4'], pool_size=3, stride=2, mode='average_exc_pad', ignore_border=False)
#    net['drop6'] = lasagne.layers.DropoutLayer(net['pool2'], p=0.5)
#    net['conv3'] = lasagne.layers.dnn.Conv2DDNNLayer(net['drop6'], num_filters=192, filter_size=3, pad=1, flip_filters=False)
#    net['cccp5'] = lasagne.layers.dnn.Conv2DDNNLayer(net['conv3'], num_filters=192, filter_size=1, flip_filters=False)
#    net['cccp6'] = lasagne.layers.dnn.Conv2DDNNLayer(net['cccp5'], num_filters=10, filter_size=1, flip_filters=False)
#    net['pool3'] = lasagne.layers.Pool2DLayer(net['cccp6'], pool_size=8, mode='average_exc_pad', ignore_border=False)
#    net['output'] = lasagne.layers.FlattenLayer(net['pool3'])
    net['input'] = lasagne.layers.InputLayer((None, 3, 32, 32),name='input')
    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'],name='conv1', num_filters=192, filter_size=5, pad=2, flip_filters=False)
    net['cccp1'] = lasagne.layers.Conv2DLayer(net['conv1'],name='cccp1', num_filters=160, filter_size=1, flip_filters=False)
    net['cccp2'] = lasagne.layers.Conv2DLayer(net['cccp1'],name='cccp2', num_filters=96, filter_size=1, flip_filters=False)
    net['pool1'] = lasagne.layers.Pool2DLayer(net['cccp2'],name='pool1', pool_size=3, stride=2, mode='max', ignore_border=False)
    net['drop3'] = lasagne.layers.DropoutLayer(net['pool1'],name='drop3', p=0.5)
    net['conv2'] = lasagne.layers.Conv2DLayer(net['drop3'],name='conv2', num_filters=192, filter_size=5, pad=2, flip_filters=False)
    net['cccp3'] = lasagne.layers.Conv2DLayer(net['conv2'],name='cccp3', num_filters=192, filter_size=1, flip_filters=False)
    net['cccp4'] = lasagne.layers.Conv2DLayer(net['cccp3'],name='cccp4', num_filters=192, filter_size=1, flip_filters=False)
    net['pool2'] = lasagne.layers.Pool2DLayer(net['cccp4'],name='pool2', pool_size=3, stride=2, mode='average_exc_pad', ignore_border=False)
    net['drop6'] = lasagne.layers.DropoutLayer(net['pool2'],name='drop6', p=0.5)
    net['conv3'] = lasagne.layers.Conv2DLayer(net['drop6'],name='conv3', num_filters=192, filter_size=3, pad=1, flip_filters=False)
    net['cccp5'] = lasagne.layers.Conv2DLayer(net['conv3'],name='cccp5', num_filters=192, filter_size=1, flip_filters=False)
    net['cccp6'] = lasagne.layers.Conv2DLayer(net['cccp5'],name='cccp6', num_filters=10, filter_size=1, flip_filters=False)
    net['pool3'] = lasagne.layers.Pool2DLayer(net['cccp6'],name='pool3', pool_size=8, mode='average_exc_pad', ignore_border=False)
    net['output'] = lasagne.layers.FlattenLayer(net['pool3'],name='output')
    return net


def __createNinCifar10__(inputLayer=None):
    if inputLayer==None:
        network = lasagne.layers.InputLayer((None,3,224,224))
    else:
        network = inputLayer
    #net['input'] = lasagne.layers.InputLayer((None, 3, 32, 32),name='input')
    network = lasagne.layers.Conv2DLayer(network,name='conv1', num_filters=192, filter_size=5, pad=2, flip_filters=False)
    network = lasagne.layers.Conv2DLayer(network,name='cccp1', num_filters=160, filter_size=1, flip_filters=False)
    network = lasagne.layers.Conv2DLayer(network,name='cccp2', num_filters=96, filter_size=1, flip_filters=False)
    network = lasagne.layers.Pool2DLayer(network,name='pool1', pool_size=3, stride=2, mode='max', ignore_border=False)
    network = lasagne.layers.DropoutLayer(network,name='drop3', p=0.5)
    network = lasagne.layers.Conv2DLayer(network,name='conv2', num_filters=192, filter_size=5, pad=2, flip_filters=False)
    network = lasagne.layers.Conv2DLayer(network,name='cccp3', num_filters=192, filter_size=1, flip_filters=False)
    network = lasagne.layers.Conv2DLayer(network,name='cccp4', num_filters=192, filter_size=1, flip_filters=False)
    network = lasagne.layers.Pool2DLayer(network,name='pool2', pool_size=3, stride=2, mode='average_exc_pad', ignore_border=False)
    network = lasagne.layers.DropoutLayer(network,name='drop6', p=0.5)
    network = lasagne.layers.Conv2DLayer(network,name='conv3', num_filters=192, filter_size=3, pad=1, flip_filters=False)
    network = lasagne.layers.Conv2DLayer(network,name='cccp5', num_filters=192, filter_size=1, flip_filters=False)
    network = lasagne.layers.Conv2DLayer(network,name='cccp6', num_filters=10, filter_size=1, flip_filters=False)
    network = lasagne.layers.Pool2DLayer(network,name='pool3', pool_size=8, mode='average_exc_pad', ignore_border=False)
    network = lasagne.layers.FlattenLayer(network,name='output')
    return network


#def createNinCifar10(inputLayer=None):
#    return __getNinCifar10ArchDict__(inputLayer)['output']


def __getVgg16ArchDict__(inputLayer=None):
    net={}
    if inputLayer==None:
        net['input'] = lasagne.layers.InputLayer((None,3,224,224))
    else:
        net['input'] = inputLayer
    net['conv1_1']= lasagne.layers.Conv2DLayer(net['input'],name='conv1_1', num_filters=64, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv1_2']= lasagne.layers.Conv2DLayer(net['conv1_1'],name='conv1_2', num_filters=64, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['pool1']=lasagne.layers.Pool2DLayer(net['conv1_2'],name='pool1',pool_size=2,stride=2,mode='max')
    net['conv2_1']= lasagne.layers.Conv2DLayer(net['pool1'],name='conv2_1', num_filters=128, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv2_2']= lasagne.layers.Conv2DLayer(net['conv2_1'],name='conv2_2', num_filters=128, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['pool2']=lasagne.layers.Pool2DLayer(net['conv2_2'],name='pool2',pool_size=2,stride=2,mode='max')

    net['conv3_1']= lasagne.layers.Conv2DLayer(net['pool2'],name='conv3_1', num_filters=256, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv3_2']= lasagne.layers.Conv2DLayer(net['conv3_1'],name='conv3_2', num_filters=256, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv3_3']= lasagne.layers.Conv2DLayer(net['conv3_2'],name='conv3_3', num_filters=256, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['pool3']=lasagne.layers.Pool2DLayer(net['conv3_3'],name='pool3',pool_size=2,stride=2,mode='max')

    net['conv4_1']= lasagne.layers.Conv2DLayer(net['pool3'],name='conv4_1', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv4_2']= lasagne.layers.Conv2DLayer(net['conv4_1'],name='conv4_2', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv4_3']= lasagne.layers.Conv2DLayer(net['conv4_2'],name='conv4_3', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['pool4']=lasagne.layers.Pool2DLayer(net['conv4_3'],name='pool4',pool_size=2,stride=2,mode='max')

    net['conv5_1']= lasagne.layers.Conv2DLayer(net['pool4'],name='conv5_1', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv5_2']= lasagne.layers.Conv2DLayer(net['conv5_1'],name='conv5_2', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['conv5_3']= lasagne.layers.Conv2DLayer(net['conv5_2'],name='conv5_3', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    net['pool5']=lasagne.layers.Pool2DLayer(net['conv5_3'],name='pool5',pool_size=2,stride=2,mode='max')
    #net['pool5']=lasagne.layers.FlattenLayer(net['pool5'])
    net['fc6']=lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(net['pool5'],p=.5),name='fc6',num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    net['fc7']=lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(net['fc6'],p=.5),name='fc7',num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    net['fc8']=lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(net['fc7'],p=.5),name='fc8',num_units=1000,nonlinearity=lasagne.nonlinearities.softmax)
    return net


def __createVgg16__(inputLayer=None):
    if inputLayer==None:
        network = lasagne.layers.InputLayer((None,3,224,224))
    else:
        network = inputLayer
    network= lasagne.layers.Conv2DLayer(network,name='conv1_1', num_filters=64, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv1_2', num_filters=64, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network=lasagne.layers.Pool2DLayer(network,name='pool1',pool_size=2,stride=2,mode='max')
    network= lasagne.layers.Conv2DLayer(network,name='conv2_1', num_filters=128, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv2_2', num_filters=128, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network=lasagne.layers.Pool2DLayer(network,name='pool2',pool_size=2,stride=2,mode='max')

    network= lasagne.layers.Conv2DLayer(network,name='conv3_1', num_filters=256, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv3_2', num_filters=256, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv3_3', num_filters=256, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network=lasagne.layers.Pool2DLayer(network,name='pool3',pool_size=2,stride=2,mode='max')

    network= lasagne.layers.Conv2DLayer(network,name='conv4_1', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv4_2', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv4_3', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network=lasagne.layers.Pool2DLayer(network,name='pool4',pool_size=2,stride=2,mode='max')

    network= lasagne.layers.Conv2DLayer(network,name='conv5_1', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv5_2', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network= lasagne.layers.Conv2DLayer(network,name='conv5_3', num_filters=512, filter_size=3, pad=1,nonlinearity=lasagne.nonlinearities.rectify ,flip_filters=False)
    network=lasagne.layers.Pool2DLayer(network,name='pool5',pool_size=2,stride=2,mode='max')
    #net['pool5']=lasagne.layers.FlattenLayer(net['pool5'])
    network=lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network,p=.5),name='fc6',num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    network=lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network,p=.5),name='fc7',num_units=4096,nonlinearity=lasagne.nonlinearities.rectify)
    network=lasagne.layers.DenseLayer(lasagne.layers.DropoutLayer(network,p=.5),name='fc8',num_units=1000,nonlinearity=lasagne.nonlinearities.softmax)
    return network



#def createVgg16(inputLayer=None):
#    return __getVgg16ArchDict__(inputLayer)['fc8']


#def loadVgg16FromCaffe(protofile,modelfile,inputLayer=None):
#    import caffe
#    archDict=__getVgg16ArchDict__(inputLayer)
#    vggCaffe=caffe.Net(protofile,modelfile,caffe.TEST)
#    transfuseCaffeWeights(archDict,vggCaffe)
#    return archDict['fc8']


#def __getDictNetArchDict__(inputLayer=None):
#    net = {}
#    if inputLayer==None:
#        net['input'] = lasagne.layers.InputLayer((None,1,32,100))
#    else:
#        net['input'] = inputLayer
#    net['conv1'] = lasagne.layers.Conv2DLayer(net['input'],name='conv1',num_filters=64,filter_size=5,stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
#    net['pool1'] = lasagne.layers.Pool2DLayer(net['conv1'],name='pool1',stride=2,pool_size=2,mode='max')
#
#    net['conv2'] = lasagne.layers.Conv2DLayer(net['pool1'],name='conv2',num_filters=128,filter_size=5,stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
#    net['pool2'] = lasagne.layers.Pool2DLayer(net['conv2'],name='pool2',stride=2,pool_size=2,mode='max')
#
#    net['conv3'] = lasagne.layers.Conv2DLayer(net['pool2'],name='conv3',num_filters=256,filter_size=3,stride=1,pad=1,nonlinearity=lasagne.nonlinearities.rectify)
#    net['conv3_5'] = lasagne.layers.Conv2DLayer(net['conv3'],name='conv3_5',num_filters=512,filter_size=3,stride=1,pad=1,nonlinearity=lasagne.nonlinearities.rectify)
#    net['pool3'] = lasagne.layers.Pool2DLayer(net['conv3_5'],name='pool3',stride=2,pool_size=2,mode='max',pad=(0,1))
#    
#    net['conv4'] = lasagne.layers.Conv2DLayer(net['pool3'],name='conv4',num_filters=512,filter_size=3,stride=1,pad=1,nonlinearity=lasagne.nonlinearities.rectify)
#    net['fc1'] = lasagne.layers.Conv2DLayer(net['conv4'],name='fc1',num_filters=4096,filter_size=(4,13),stride=1,nonlinearity=lasagne.nonlinearities.rectify,pad=0)
#    net['fc2'] = lasagne.layers.Conv2DLayer(net['fc1'],name='fc2',num_filters=4096,filter_size=(1,1),stride=1,nonlinearity=lasagne.nonlinearities.rectify,pad=0)
#    #lasagne doesn't allow softmax on convnets
#    #net['fc_class'] = lasagne.layers.Conv2DLayer(net['fc2'],name='fc_class',num_filters=88172,filter_size=(1,1),stride=1,nonlinearity=lasagne.nonlinearities.rectify,pad=0)
#    #net['fc_out']
#    net['fc_class'] = lasagne.layers.Conv2DLayer(net['fc2'],name='fc_class',num_filters=88172,filter_size=(1,1),stride=1,nonlinearity=lasagne.nonlinearities.identity,pad=0)
#    net['fc_reshaped'] = lasagne.layers.ReshapeLayer(net['fc_class'],(-1,88172),name='fc_reshaped')
#    net['fc_out']=lasagne.layers.NonlinearityLayer(net['fc_reshaped'],name='fc_out',nonlinearity=lasagne.nonlinearities.softmax)
#    return net


def __createDictNet__(inputLayer=None):
    if not hasattr(__createDictNet__, 'name'):
        __createDictNet__.name = 'DictNet'
    if inputLayer==None:
        network = lasagne.layers.InputLayer((None,1,32,100))
    else:
        network = inputLayer
    network = lasagne.layers.Conv2DLayer(network,name='conv1',num_filters=64,filter_size=5,stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Pool2DLayer(network,name='pool1',stride=2,pool_size=2,mode='max')

    network = lasagne.layers.Conv2DLayer(network,name='conv2',num_filters=128,filter_size=5,stride=1,pad=2,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Pool2DLayer(network,name='pool2',stride=2,pool_size=2,mode='max')

    network = lasagne.layers.Conv2DLayer(network,name='conv3',num_filters=256,filter_size=3,stride=1,pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network,name='conv3_5',num_filters=512,filter_size=3,stride=1,pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Pool2DLayer(network,name='pool3',stride=2,pool_size=2,mode='max',pad=(0,1))
    
    network = lasagne.layers.Conv2DLayer(network,name='conv4',num_filters=512,filter_size=3,stride=1,pad=1,nonlinearity=lasagne.nonlinearities.rectify)
    network = lasagne.layers.Conv2DLayer(network,name='fc1',num_filters=4096,filter_size=(4,13),stride=1,nonlinearity=lasagne.nonlinearities.rectify,pad=0)
    network = lasagne.layers.Conv2DLayer(network,name='fc2',num_filters=4096,filter_size=(1,1),stride=1,nonlinearity=lasagne.nonlinearities.rectify,pad=0)
    #lasagne doesn't allow softmax on convnets
    #network = lasagne.layers.Conv2DLayer(network,name='fc_class',num_filters=88172,filter_size=(1,1),stride=1,nonlinearity=lasagne.nonlinearities.rectify,pad=0)
    #return network
    network = lasagne.layers.Conv2DLayer(network,name='fc_class',num_filters=88172,filter_size=(1,1),stride=1,nonlinearity=lasagne.nonlinearities.identity,pad=0)
    network = lasagne.layers.ReshapeLayer(network,(-1,88172),name='fc_reshaped')
    network=lasagne.layers.NonlinearityLayer(network,name='fc_out',nonlinearity=lasagne.nonlinearities.softmax)
    return network


#def createDictNet(inputLayer=None):
#    return __getDictNetArchDict__(inputLayer)['fc_out']


def create(modelName,**kwargs):
    modelDict={'dictnet':__createDictNet__,'lenet':createLenet,'nincifar10':__createNinCifar10__,'vgg16':__createVgg16__}
    return modelDict[modelName.lower()](**kwargs)


def loadCaffe(modelName,**kwargs):
    modelName=modelName.lower()
    caffeModelPath='/home/anguelos/models/'
    caffeFiles={
    'dictnet':('jaderberg_text_models/deploy_dictnet.prototxt',
               'jaderberg_text_models/jaderberg_dictnet_orig_imported.caffemodel'),
    'vgg16':('vgg/VGG_ILSVRC_16_layers_deploy.prototxt',
               'vgg/VGG_ILSVRC_16_layers.caffemodel'),
    'nincifar10':('nin_cifar10/nin_cifar10.prototxt','nin_cifar10/nin_cifar10_deploy.model')
                }
    import caffe
    lasagneModel=create(modelName,**kwargs)
    caffeModel=caffe.Net(caffeModelPath+caffeFiles[modelName][0],caffeModelPath+caffeFiles[modelName][1],caffe.TEST)
    transfuseCaffeWeightsSimple(lasagneModel,caffeModel)
    return lasagneModel


#def loadDictNetFromCaffe(protofile,modelfile,inputLayer=None):
#    import caffe
#    archDict=__getDictNetArchDict__(inputLayer)
#    dictnetCaffe=caffe.Net(protofile,modelfile,caffe.TEST)
#    transfuseCaffeWeights(archDict,dictnetCaffe)
#    return archDict['fc_out']
#
#
#def loadDictNetFromCaffeSimple(protofile,modelfile,inputLayer=None):
#    import caffe
#    model=__getDictNetArchDict__(inputLayer)['fc_class']
#    dictnetCaffe=caffe.Net(protofile,modelfile,caffe.TEST)
#    transfuseCaffeWeightsSimple(model,dictnetCaffe)
#    return model
