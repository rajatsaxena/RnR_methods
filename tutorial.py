# -*- coding: utf-8 -*-
"""
Created on Tue May 12 13:27:29 2020

@author: Rajat
"""

import utils
import numpy as np
import scipy.stats as scst
import scipy.ndimage as scnd
import matplotlib.pyplot as plt

# initial parameters
numCells = 100 # of place fields for simulations
numBins = 101
numIterations = 100

# first, let's make some place fields.
# 1. Dupret style, reward/end over-representation
# 2. linearly spaced PFs
# 3. randomly spaced PFs
offsets_rate = np.zeros((3, numCells))
offsets_rate[0,:] = np.array(utils.sigmoid(np.arange(100),50,0.09)*100)
offsets_rate[1,:] = np.arange(100) 
offsets_rate[2,:] = np.random.uniform(0,100,100)
offsets_rate = np.array(offsets_rate, dtype='int')

# array to hold rateMap information
rateMaps = np.zeros((len(offsets_rate), numCells, numBins))
# create the rate maps
for o in range(len(offsets_rate)):
    for neuron in range(numCells):
        rm = scnd.gaussian_filter1d(np.concatenate((np.zeros(offsets_rate[o][neuron]+1), np.array([1]), np.zeros(99-offsets_rate[o][neuron]))),5)
        rateMaps[o,neuron,:] = rm

# ok, now let's make some ripple examples..
orip1 = [np.concatenate((np.ceil(numCells/2. - np.logspace(2,0.8,50)/2+3),np.ceil(np.logspace(0.8,2,50)/2. + numCells/2. - 3)))]
orip2 = [np.arange(100)]
offsets_rip = np.concatenate((orip1, orip2),0)

# variables to hold different quality metrics
integral = np.zeros((len(offsets_rip),len(offsets_rate), numIterations))
integral_cellId_shuf = np.zeros((len(offsets_rip),len(offsets_rate), numIterations))
rankOrder = np.zeros((len(offsets_rip), len(offsets_rate)))
rankOrderFirstSpk = np.zeros((len(offsets_rip), len(offsets_rate)))
integral_shuffle = np.zeros((len(offsets_rip), len(offsets_rate), numIterations))
rankOrder_shuf = np.zeros((len(offsets_rip), len(offsets_rate), numIterations))
rankOrderFirstSpk_shuf = np.zeros((len(offsets_rip), len(offsets_rate), numIterations))
rankOrder_cellId_shuf = np.zeros((len(offsets_rip), len(offsets_rate), numIterations))
rankOrderFirstSpk_cellId_shuf = np.zeros((len(offsets_rip), len(offsets_rate), numIterations))
# generate ripple event          
rippleEvent = np.zeros((len(offsets_rip),numCells,numBins))
# get stats for each ripple event while generating it
for o in range(len(offsets_rip)):
    for oo in range(len(offsets_rate)):
        # create rippl events
        for neuron in range(numCells):
            rip_ = np.concatenate((np.zeros(int(offsets_rip[o][neuron]+1)), np.array([1]), np.zeros(int(99-offsets_rip[o][neuron]))))
            rippleEvent[o,neuron,:] = rip_
            
        # generate ripple event to now check for template matching 
        for iteration in range(numIterations):
            spks = np.where(rippleEvent[o]==1)[0]
            rip = np.copy(rippleEvent[o])
            r = np.random.permutation(100)
            rip[spks[r[:80]]] = 0
            
            # radon transform
            Pr, prMax = utils.placeBayes(rip.T, rateMaps[oo], 1)
            slope, integral[o,oo][iteration] = utils.Pr2Radon(Pr.T)
            
            # linear weighted correlation
            
            # rank-order correlations
            #  only calculate once for the actual data
            if iteration == 0:
                _, _, ord1 = utils.sort_cells(rateMaps[oo])
                _, _, ord2 = utils.sort_cells(rippleEvent[o])
                _, ord_firstSpk = utils.sort_rows(rippleEvent[o],order='descending')
                rankOrder[o,oo], _ = scst.spearmanr(ord1,ord2)
                rankOrderFirstSpk[o,oo], _ = scst.spearmanr(ord1,ord_firstSpk)
            
            # shuffle cellID
            shuf = utils.shuffleCellID(rateMaps[oo]) 
            [Pr, prMax] = utils.placeBayes(rip.T, shuf, 1); 
            Pr[np.isnan(Pr)] = 0
            #[bayesLinearWeighted_cellID_shuf(event),outID] = makeBayesWeightedCorr1(Pr,ones(size(Pr,1),1));
            _, integral_cellId_shuf[o,oo][iteration] = utils.Pr2Radon(Pr.T)
            _, _, ord_shuf = utils.sort_cells(shuf)
            _, ord_firstSpkshuf = utils.sort_rows(shuf)
            rankOrder_cellId_shuf[o,oo][iteration], _ = scst.spearmanr(ord_shuf,ord2)
            rankOrderFirstSpk_cellId_shuf[o,oo][iteration], _ = scst.spearmanr(ord_firstSpkshuf,ord2)
        
            # shuffle circular
            shuf = utils.shuffleCircular(rateMaps[oo])
            Pr, _ = utils.placeBayes(rip.T,shuf,1)
            _, integral_shuffle[o,oo][iteration] = utils.Pr2Radon(Pr.T) 
            _, _, ord_shuf = utils.sort_cells(shuf)
            _, ord_firstSpkshuf = utils.sort_rows(shuf)
            rankOrder_shuf[o,oo][iteration], _ = scst.spearmanr(ord_shuf,ord2)
            rankOrderFirstSpk_shuf[o,oo][iteration], _ = scst.spearmanr(ord_firstSpkshuf,ord2)

# Plotting results
conditions = len(offsets_rate)*len(offsets_rip)
cond = 1
for o in range(len(offsets_rip)):
    for oo in range(len(offsets_rate)):
        plt.subplot(conditions,4,cond*4-3)
        plt.imshow(rateMaps[oo])
        plt.xlabel('position')
        plt.ylabel('neuron #')
        if cond==1: plt.title('Behavior template')
        
        plt.subplot(conditions,4,cond*4-2)
        plt.imshow(rippleEvent[o])
        if cond==1: plt.title('ripple template')
        plt.xlabel('time bin')
        
        plt.subplot(conditions,4,cond*4-1)
        plt.hist(integral[o,oo],np.arange(0,0.05,0.001), alpha=0.75)
        plt.hist(integral_shuffle[o,oo],np.arange(0,0.05,0.001), alpha=0.75)   
        plt.hist(integral_cellId_shuf[o,oo],np.arange(0,0.05,0.001), alpha=0.75)   
        if cond==1: plt.title('radon integral')
        plt.xlim([0,0.03])
        
        plt.subplot(conditions,4,cond*4)
        plt.plot([rankOrder[o,oo], rankOrder[o,oo]],[0,50],'r')
        plt.plot([rankOrderFirstSpk[o,oo], rankOrderFirstSpk[o,oo]],[0,50],'m')
        plt.hist(rankOrder_shuf[o,oo], alpha=0.75)
        plt.hist(rankOrder_cellId_shuf[o,oo], alpha=0.75)
        plt.hist(rankOrderFirstSpk_shuf[o,oo], alpha=0.75)
        plt.hist(rankOrderFirstSpk_cellId_shuf[o,oo], alpha=0.75)
        if cond==1: plt.title('rank order correlation')
        plt.xlim([-1,1])
        
        cond = 1+cond
# plt.tight_layout()
plt.show()