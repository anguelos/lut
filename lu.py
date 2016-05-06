#!/usr/bin/env python


import sys
import os
import time
import cPickle
import random

import numpy as np
import theano
import theano.tensor as T

import lasagne

from matplotlib import pyplot as plt

import lasagne.regularization
from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams
from lasagne.random import get_rng


from core import *















#def sharedCopy(inputLayer,sl):
#    if type(sl)==lasagne.layers.Conv2DLayer:
#        return lasagne.layers.Conv2DLayer(inputLayer,num_filters=sl.num_filters,filter_size=sl.filter_size,nonlinearity=sl.nonlinearity,pad=sl.pad)
#    elif type(sl)==lasagne.layers.Pool2DLayer():
#        pass

















