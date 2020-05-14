# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:33:20 2020

@author: Rajat
"""
import sys
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from skimage.transform import radon

"""
matlab style gaussian function
1/1+exp^(-a*(x-c))
"""
def sigmoid(x,c,a):
    return 1. / (1 + np.exp(-a*(x-c)))

"""
Inputs:
Cr = [nTemporalBin X nCell] matrix of binned firing rates
rateMap = [nSpatialBin X nCell] firing rate 'template'
binLength = scalar of the duration of the bins in 'Cr'

Outputs:
Pr = [nTemporalBin X nSpatialBins] matrix of posterior probabilities
prMax = the spatial bin with higher spatial probabilities for each
  temporalBin in Cr
"""
def placeBayes(Cr, rateMap, binLength):
    Cr = Cr*binLength
    rateMap = rateMap.T
    term2 = np.exp((-1)*binLength*np.sum(rateMap.T,0))
    mp = 1./rateMap.shape[1]
    Pr = []
    
    c = np.repeat(Cr[:, :, np.newaxis], rateMap.shape[0], axis=2)
    b = np.repeat(rateMap.T[:, :, np.newaxis], c.shape[0], axis=2)
    b = np.moveaxis(b, -1, 0)
    
    u = mp*np.prod(b**c, 1)
    Pr = u*np.repeat(term2[:,np.newaxis], u.shape[0], axis=1).T
    Pr = Pr/np.repeat(np.sum(Pr,1)[:,np.newaxis], Pr.shape[1], axis=1)
    
    m = np.argmax(Pr,1)
    prMax = m.T
    
    if np.sum(np.isinf(Pr))>0:
        sys.error('Do Not Approach the Infitnite')
    return Pr, prMax

def fPolyFit(x,y,n):
    V = np.ones((len(x),n+1))
    for j in np.arange(1,0,-1):
        V[:,j-1] = V[:,j]*x
    # Solve least squares problem
    Q, R = sp.linalg.qr(V, mode='economic')
    y = np.reshape(y, (len(y),1))
    p =  np.transpose(np.linalg.inv(R).dot(Q.T.dot(y)))[0]
    return p

"""
computes max projection line, using radon transform, on Pr matrix
"""
def Pr2Radon(Pr, plotting=0):
    Pr[np.isnan(Pr)] = 0
    theta = np.arange(0,180+0.5,0.5)
    R = radon(Pr,theta,circle=False)
    bw_ = R.shape[0]//2
    xp = np.linspace(-bw_,bw_,R.shape[0])
    
    y=[None, None]
    x=[None, None]
    y[0] = np.floor((Pr.shape[0])//2.)
    x[0] = np.floor((Pr.shape[1])//2.)
    
    I = np.nanargmax(R,0)
    Y = R[I]
    
    locs = np.arange(len(Y))
    slope = np.zeros(len(locs))
    integral = np.zeros(len(locs))
    curve = np.zeros(Pr.shape[1])
    for pk in range(len(locs)):
        angle = theta[locs[pk]]
        offset = xp[I[locs[pk]]]
        if offset==0:
            offset=0.01

        y[1] = y[0] + offset*np.sin(np.deg2rad(-angle))
        x[1] = x[0] + offset*np.cos(np.deg2rad(-angle))
        coeffs = fPolyFit(x, y, 1)
        xx = np.arange(Pr.shape[1])
        yy = (-1/coeffs[0])*(xx - x[0]) + y[0] - offset
        coeffs = fPolyFit(xx, yy, 1)
        slope[pk] = coeffs[0]
        
        # rise/run limit to calc integral (must be in the frame)
        if abs(slope[pk]) < 2*Pr.shape[0]/Pr.shape[1] and abs(slope[pk]) > 1.5:
            for i in range(len(xx)):
                if yy[i] > .5 and yy[i] < Pr.shape[0] - .5:                    
                    curve[i] = Pr[int(yy[i]),int(xx[i])]
                else:
                    curve[i] = np.nan
            integral[pk] = np.nanmean(curve)
        else:
            integral[pk] = np.nan
            slope[pk] = np.nan
    
    # weird typecasting fix
    integral = np.array(integral, dtype='float64')
    idx = np.nanargmax(integral)
    integral = integral[idx]
    slope = slope[idx]    
    
    return slope, integral


"""
Sorts two matrices by ordering the maximums of the first and using that
order to rearrange both
#1 = max, #2 = min
"""
def sort_cells(item,item2=None,num=1):
    if item2 is None:
        item2 = np.copy(item)
     
    new_item = np.zeros((item.shape[0], item.shape[1]))
    new_item2 = np.zeros((item2.shape[0], item2.shape[1]))
    
    if num==1:
        d = np.argmax(item, axis=1)
        ddd = np.argsort(d)
        new_item = item[ddd,:]
        new_item2 = item2[ddd,:] 
        order = ddd
        
    if num==2:
        d = np.argmin(item, axis=1)
        ddd = np.argsort(d)
        new_item = item[ddd,:]
        new_item2 = item2[ddd,:] 
        order = ddd
        
    return new_item, new_item2, order

"""
circularly shift each row
"""
def shuffleCircular(data):
    data1 = np.copy(data)
    for i in range(data1.shape[0]):
        shift_ = np.random.randint(0, data1.shape[1], 1)
        shift_ = shift_[0]
        data1[i,:] = np.roll(data1[i,:], shift_)
    return data1