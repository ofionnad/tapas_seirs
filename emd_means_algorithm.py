#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 8 14:50:50 2023
Author: @ofionnad

"""

import numpy as np
from ot import emd2
from sklearn.cluster import KMeans
from scipy import stats
from scipy import integrate
import geopandas as gpd

# Use the ecdf data from the NDVI differences
# Must be in the shape (x, y)
tzo = np.load('treatment_zone_observed.pkl')
czo = np.load('control_zone_observed.pkl')

shp = gpd.read_file('/home/dualta/Documents/tapas/seirs_data/shapefiles/irrigation_shapefile_africarice/SRV_irr_v10/SRV_irr_v10.shp')

control_sites = czo.within(shp[shp['F2']=='SN0011'])
treatment_sites = tzo.within(shp[shp['F2']=='SN1358'])

# Define a custom distance function using the EMD
def emd_distance(x, y):
    return emd2(x, y, np.ones(x.shape[0]), np.ones(y.shape[0]))

# define the custom wasserstein distance metric
def wasserstein_distance_metric(x, y):
    x_ecdf = np.sort(x)
    y_ecdf = np.sort(y)
    return stats.wasserstein_distance(x_ecdf, y_ecdf)


# Initialize the K-Means model
kmeans = KMeans(n_clusters=5, init='k-means++', max_iter=300, n_init=10, algorithm='elkan', distance_function=emd_distance)

# Fit the model to the data
kmeans.fit(data)

# Get the cluster assignments
clusters = kmeans.predict(data)

# Get the cluster centers
centers = kmeans.cluster_centers_


def dist2_wasserstein(elist1, elist2, p):
    M = len(elist1)
    N = len(elist2)
    
    trflst = elist_fform(np.concatenate((elist1, elist2)))
    flist1 = trflst['fval'][:M]
    flist2 = trflst['fval'][M:(M+N)]
    
    qseq = np.linspace(1e-6, 1-(1e-6), 8128)
    quants1 = [] # compute quantile functions first
    quants2 = []
    for i in range(M):
        quants1.append(np.asarray(stats.mstats.mquantiles(elist1[i], qseq)))
    for j in range(N):
        quants2.append(np.asarray(stats.mstats.mquantiles(elist2[j], qseq)))
   
    output = np.zeros((M, N))
    for i in range(M):
        vali = quants1[i]
        for j in range(N):
            valj = quants2[j]
            valij = np.abs(vali - valj)
            if (p == np.inf):
                output[i][j] = np.max(valij)
            else:
                output[i][j] = ((integrate.simps(valij**p, x=qseq))**(1/p))
    return output

def elist_fform(elist):
    nlist = len(elist)
    # compute knot points
    allknots = np.zeros((nlist, 2))
    for i in range(nlist):
        tgt = stats.mstats.knots(elist[i])
        allknots[i, :] = (np.min(tgt), np.max(tgt))
    
    mint = np.min(allknots[:, 0]) - 0.01
    maxt = np.max(allknots[:, 1]) + 0.01
    ssize = min((maxt-mint)/1000, 0.001)
    tseq = np.arange(mint, maxt+ssize, ssize)
    # return the list of y values
    outY = []
    for i in range(nlist):
        tgt = elist[i]
        outY.append(tgt(tseq))
    # return the result
    output = dict()
    output["tseq"] = tseq
    output["fval"] = outY # list of function values
    return output