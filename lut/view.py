# -*- coding: utf-8 -*-
"""
Created on Sun Mar 20 18:43:16 2016

@author: anguelos
"""

from core import getActivationsAsFuction
from core import getLayerDict
import numpy as np

from matplotlib import pyplot as plt
def getModelArchStr(model):
    d=getLayerDict(model)
    res=('| L#        |%-34s  |%-25s  |%-25s|\n'%('Layer type','Activation Sizes','Weight Sizes'))
    res='| L#         | Layer Type                        | Activation Size           | Weights Size            |\n'
    res='_'*len(res[:-1])+'\n'+res+'-'*len(res[:-1])+'\n'
    res+=('| In         |%-33s  |%-25s  |%-25s|\n'%(str(type(d['input'])).split('.')[-1][:-2],str(d['input'].shape),'NA'))
    #prevShape=d['input'].shape
    for pos in range(d['sz']):
        newShape=d[pos].output_shape
#        if type(d[pos])==lasagne.layers.InverseLayer:
#            newShape=d[pos].input_layers[-1].get_output_shape_for(prevShape)
#        else:
#            newShape=d[pos].get_output_shape_for(prevShape)
        try:
            weightSizes=str(d[pos].get_params()[0].get_value().shape)
        except:
            weightSizes='NA'
        name=str(d[pos].name)
        name=name[:min([len(name),8])]
        res+=('|%3d %8s|%-33s  |%-25s  |%-25s|\n'%(pos,name,str(type(d[pos])).split('.')[-1][:-2],str(newShape),str(weightSizes)))
        #prevShape=newShape
    res+='-'*len(res.split('\n')[0])+'\n'
    return res

def plotActivations(model,layer,img=None,**kwargs):
    p={'toImg':False,'avg':True,'normFilter':(0,1),'boxTiltes':True,'grid':False,'maxOutput':40,'crop':(0,0),'show':(lambda :plt.show())}
    p.update(kwargs)
    if img==None:
        inputVar=getLayerDict(model)['input']
        sz=list(inputVar.shape)
        sz[0]=1
        img=np.zeros(tuple(sz))
        img[sz[0]/2,sz[1]/2,sz[2]/2,sz[3]/2]=1
    f=getActivationsAsFuction(model,layer)
    out=f(img)
    if p['avg']:
        out=out.mean(axis=0,keepdims=True)
        img=img.mean(axis=0,keepdims=True)
    else:
        out=out[:1,:,:,:]
    nbPlots=min(p['maxOutput'],out.shape[1]+1)
    gridWidth=1+nbPlots/(int(nbPlots**.5))
    gridHeight=1+nbPlots/gridWidth
    f, axarr = plt.subplots(gridHeight,gridWidth)
    cropImg=img[0,0,p['crop'][0]:img.shape[2]-p['crop'][0],p['crop'][1]:img.shape[3]-p['crop'][1]]
    out=out[:,:,p['crop'][0]:out.shape[2]-p['crop'][0],p['crop'][1]:out.shape[3]-p['crop'][1]]
    axarr[0,0].imshow(cropImg,interpolation='nearest',cmap='gray')
    if p['boxTiltes']:
        axarr[0,0].set_title('Input Image')
    if p['grid']:
        xl=axarr[0,0].get_xlim();axarr[0,0].set_xticks(np.arange(xl[0],xl[1]+.5,1))
        xl=axarr[0,0].get_ylim();xl=sorted(xl);axarr[0,0].set_yticks(np.arange(xl[0],xl[1]+.5,1))
        axarr[0,0].grid(which='major', axis='both', linestyle='-',linewidth=1,color='green',alpha=.125)
    plt.setp(axarr[0,0].get_xticklabels(), visible=False);plt.setp(axarr[0,0].get_yticklabels(), visible=False)
    for k in range(1,nbPlots):
        if p['normFilter']==(0,1):
            cMin=0;cMax=1
        elif p['normFilter']=='filter':
            cMin=out[0,-(k-1),:,:].min();cMax=out[0,-(k-1),:,:].max()
        elif p['normFilter']=='all':
            cMin=out.min();cMax=out.max()
        ax=axarr[k%gridHeight,k/gridHeight]
        if p['boxTiltes']:
            ax.set_title('Filter %d'%(k-1))
        ax.imshow(out[0,-(k-1),:,:],interpolation='nearest',cmap='gray', vmin=cMin, vmax=cMax)
        if p['grid']:
            xl=ax.get_xlim();ax.set_xticks(np.arange(xl[0],xl[1]+.5,1))
            xl=ax.get_ylim();xl=sorted(xl);ax.set_yticks(np.arange(xl[0],xl[1]+.5,1))
            ax.grid(which='major', axis='both',linestyle='-',linewidth=1,color='red',alpha=.125)
        plt.setp(ax.get_xticklabels(), visible=False);plt.setp(ax.get_yticklabels(), visible=False)
    for k in range(nbPlots,gridWidth*gridHeight):
        plt.setp(axarr[k%gridHeight,k/gridHeight], visible=False)
    plt.grid(True)
    p['show']()

def view2dProjection(ds,**kwargs):
    p={'projection':'pca','model':None,'fname':None,'stabiliseBy':None}
    p.update(kwargs)
    dimReduction={'pca':__reducePca__,'tsne':__reduceTsne__}
    labels=np.unique(ds['yLabs'])
    colors=[]
    n=256
    while len(colors)<(len(labels)):
        for zero,one in [(k,k+n-1) for k in range(0,256,n)]:
            colors.append(('#%02X%02X%02X'%(one,zero,zero)))
            colors.append(('#%02X%02X%02X'%(zero,one,zero)))
            colors.append(('#%02X%02X%02X'%(zero,zero,one)))
            colors.append(('#%02X%02X%02X'%(one,one,zero)))
            colors.append(('#%02X%02X%02X'%(zero,one,one)))
            colors.append(('#%02X%02X%02X'%(one,zero,one)))
        n=n/2
    if p['model']:
        f=getActivationsAsFuction(p['model'])
        x=f(ds['X'])
    else:
        x=ds['X']
    if len(x.shape)>2:
        x=x.reshape([-1,np.prod(x.shape[1:])])
    x=dimReduction[p['projection']](x)
    y=ds['yLabs']
    if p['stabiliseBy']!=None:
        xS=p['stabiliseBy']
        xS=xS.reshape([-1,np.prod(xS.shape[1:])])
        xS=__reducePca__(xS,2)
        if np.correlate(xS[:,0],x[:,0])<0:
            x[:,0]*=-1
        if np.correlate(xS[:,1],x[:,1])<0:
            x[:,1]*=-1
    plt.clf()
    plt.plot([-1,1,1,-1,-1],[-1,-1,1,1,-1],'k')
    plt.xlim([-10,10])
    plt.ylim([-10,10])
    for k in range(len(labels)):
        plt.plot(x[y==labels[k],0],x[y==labels[k],1],color=colors[k],marker='.',markersize=2,linewidth=0)
    if p['fname']:
        plt.savefig(p['fname'])
    else:
        plt.show()

def __reducePca__(X = np.array([]), no_dims = 2):
    """Runs PCA on the NxD array X in order to reduce its dimensionality to no_dims dimensions."""
    #print "Preprocessing the data using PCA..."
    (n, d) = X.shape;
    X = X - np.tile(np.mean(X, 0), (n, 1));
    (l, M) = np.linalg.eig(np.dot(X.T, X));
    M=M[:,0:no_dims]
    Y = np.dot(X, M);
    return Y;


def __reduceTsne__(X = np.array([]), no_dims = 2, initial_dims = 10, perplexity = 30.0):
    """Runs t-SNE on the dataset in the NxD array X to reduce its dimensionality to no_dims dimensions.
    The syntaxis of the function is Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    adapted from http://lvdmaaten.github.io/tsne/, by Laurens van der Maaten."""
    def Hbeta(D = np.array([]), beta = 1.0):
        """Compute the perplexity and the P-row for a specific value of the precision of a Gaussian distribution."""
        # Compute P-row and corresponding perplexity
        P = np.exp(-D.copy() * beta);
        sumP = sum(P);
        H = np.log(sumP) + beta * np.sum(D * P) / sumP;
        P = P / sumP;
        return H, P;
    
    def x2p(X = np.array([]), tol = 1e-5, perplexity = 30.0):
        """Performs a binary search to get P-values in such a way that each conditional Gaussian has the same perplexity."""
        # Initialize some variables
        #print "Computing pairwise distances..."
        (n, d) = X.shape;
        sum_X = np.sum(np.square(X), 1);
        D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X);
        P = np.zeros((n, n));
        beta = np.ones((n, 1));
        logU = np.log(perplexity);
        # Loop over all datapoints
        for i in range(n):
            # Print progress
            if i % 500 == 0:
                print "Computing P-values for point ", i, " of ", n, "..."
            # Compute the Gaussian kernel and entropy for the current precision
            betamin = -np.inf;
            betamax =  np.inf;
            Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))];
            (H, thisP) = Hbeta(Di, beta[i]);
            # Evaluate whether the perplexity is within tolerance
            Hdiff = H - logU;
            tries = 0;
            while np.abs(Hdiff) > tol and tries < 50:
                # If not, increase or decrease precision
                if Hdiff > 0:
                    betamin = beta[i].copy();
                    if betamax == np.inf or betamax == -np.inf:
                        beta[i] = beta[i] * 2;
                    else:
                        beta[i] = (beta[i] + betamax) / 2;
                else:
                    betamax = beta[i].copy();
                    if betamin == np.inf or betamin == -np.inf:
                        beta[i] = beta[i] / 2;
                    else:
                        beta[i] = (beta[i] + betamin) / 2;
                # Recompute the values
                (H, thisP) = Hbeta(Di, beta[i]);
                Hdiff = H - logU;
                tries = tries + 1;
            # Set the final row of P
            P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP;
        # Return final P-matrix
        print "Mean value of sigma: ", np.mean(np.sqrt(1 / beta));
        return P;
    # Check inputs
    if isinstance(no_dims, float):
        print "Error: array X should have type float.";
        return -1;
    if round(no_dims) != no_dims:
        print "Error: number of dimensions should be an integer.";
        return -1;
    # Initialize variables
    X = __reducePca__(X, initial_dims).real;
    (n, d) = X.shape;
    max_iter = 100;
    initial_momentum = 0.5;
    final_momentum = 0.8;
    eta = 500;
    min_gain = 0.01;
    Y = np.random.randn(n, no_dims);
    dY = np.zeros((n, no_dims));
    iY = np.zeros((n, no_dims));
    gains = np.ones((n, no_dims));
    # Compute P-values
    P = x2p(X, 1e-5, perplexity);
    P = P + np.transpose(P);
    P = P / np.sum(P);
    P = P * 4;                                    # early exaggeration
    P = np.maximum(P, 1e-12);
    # Run iterations
    for iter in range(max_iter):
        # Compute pairwise affinities
        sum_Y = np.sum(np.square(Y), 1);
        num = 1 / (1 + np.add(np.add(-2 * np.dot(Y, Y.T), sum_Y).T, sum_Y));
        num[range(n), range(n)] = 0;
        Q = num / np.sum(num);
        Q = np.maximum(Q, 1e-12);
        # Compute gradient
        PQ = P - Q;
        for i in range(n):
            dY[i,:] = np.sum(np.tile(PQ[:,i] * num[:,i], (no_dims, 1)).T * (Y[i,:] - Y), 0);
        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0) != (iY > 0)) + (gains * 0.8) * ((dY > 0) == (iY > 0));
        gains[gains < min_gain] = min_gain;
        iY = momentum * iY - eta * (gains * dY);
        Y = Y + iY;
        Y = Y - np.tile(np.mean(Y, 0), (n, 1));
        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q));
            print "Iteration ", (iter + 1), ": error is ", C
        # Stop lying about P-values
        if iter == 100:
            P = P / 4;
    # Return solution
    return Y;

