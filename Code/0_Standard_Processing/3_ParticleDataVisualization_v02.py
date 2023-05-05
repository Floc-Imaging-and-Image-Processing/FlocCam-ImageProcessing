#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:51:10 2022

@author: strom-adm
"""

#user inputs
#............................................................................

# directory to process
where = '0_Paths.csv' # manage paths with the 0_Paths.csv file if you don't run the file locally

vdist = 1 # use particle vol for distributions rather than the frequency weighting... if 1, then the w value below does not matter, if 0, w value is used
w = 3 # the distribution weighting value 0 = by number, 1 = by diameter (Ali's), 2 = by area, 3 = by vol

nb = 30 # number of bins
useLISST = 1
lisstbinsfile = '../LISST_bins/LISST_bins_sphere.csv' # LISST_bins_random.csv

combine = 'no' # average over multiple output PSDs (or minutes). Choose "yes" and then start and end of averaging period. Choose 'no' to only look at the minute of interest (line above)
minfirst = 1
minlast = 18

minoi = 1 # individual distribution (minute of interest).

datafile = 'd_mu.csv'
output_folder = '0_analysis_output' # make sure this directory does not exist (created directory will be main path/output_folder)

#............................................................................

# imports

import numpy as np
import os
import pandas as pd
import matplotlib.pyplot as plt
import time
import sys
time1 = time.time()

# find the data directories and setup an output direcectory

CodePath = os.getcwd()

if where == 'local':
    path_main = os.getcwd()
else:
    paths = pd.read_csv(where)

for j in range(1,len(paths)):
    path_main = paths.iloc[j,1]

    os.chdir(path_main)

    super_path = os.path.join(path_main,output_folder)

    if os.path.isdir(super_path) != True:
        print('you need to process the data first with 2_ParticleDataProcess_XX.py')
        sys.exit()

    # pull in LISST bin sizes if using

    if useLISST == 1:
        lisst_bins = pd.read_csv(CodePath+'/'+lisstbinsfile)
        edges = list(lisst_bins.Lower)
        edges.append(lisst_bins.Upper.iloc[-1])

    # read in the data file

    d_mu = pd.read_csv(super_path+'/'+datafile)

    if combine == 'yes':
        subset=d_mu.iloc[:,minfirst-1:minlast+1]
        d=np.array(subset.stack(dropna=True))
    else:
        d=np.array(d_mu.iloc[:,minoi-1])
        d=d[~np.isnan(d)]

    # look at individual distributions

    dlog=np.log(d)

    if vdist == 1:
        dvol_uL = 1e-9*(np.pi/6)*d**3 # volume in microliters (1 micron^3 = 1e-9 microliters)
        if useLISST == 1:
            values, base = np.histogram(d, bins=edges, weights=dvol_uL)
            centers = np.array(lisst_bins.Median)
            width = base[1:]-base[:-1]
        else:
            values, logbase = np.histogram(dlog, bins=nb, weights=dvol_uL)
            logcenters = (logbase[1:]+logbase[:-1])/2
            base = np.exp(logbase)
            width = base[1:]-base[:-1]
            centers = np.exp(logcenters)
        print('total vol of sediment [$\mu L$] =',sum(values))
    else:
        if useLISST == 1:
            values, base = np.histogram(d, bins=edges)
            centers = np.array(lisst_bins.Median)
            width = base[1:]-base[:-1]
            values=values*(centers)**w
        else:
            values, logbase = np.histogram(dlog, bins=nb)
            logcenters = (logbase[1:]+logbase[:-1])/2
            base = np.exp(logbase)
            width = base[1:]-base[:-1]
            centers = np.exp(logcenters)
            values=values*(centers)**w

    cumulative = np.cumsum(values) # adds up the frequencies (total is the total number of points)
    valuefrac = values/np.sum(values)
    perc = cumulative/cumulative[len(cumulative)-1]
    percfiner = perc
    percfiner = np.insert(percfiner,0,0)
    d16=np.interp(0.16,percfiner,base)
    d50=np.interp(0.5,percfiner,base)
    d84=np.interp(0.84,percfiner,base)

    if useLISST == 1:
        binmethod = "LISST"
    else:
        binmethod = "auto"


    if vdist == 1:
        ylab1 = 'Volume Frac [$\mu L/\mu L$]'
        ylab2 = 'Fraction Finer [by vol]'
        data1 = 'vol_muL'
    else:
        ylab1 = 'Frequency Frac [$d^{'+str(w)+'}$ weighting]'
        ylab2 = 'Fraction Finer [$d^{'+str(w)+'}$ weighting]'
        data1 = 'number'

    if combine == 'yes':
        dist = str(minfirst)+'-'+str(minlast)
    else:
        dist = str(minoi)

    # plot the histogram
    fig=plt.figure()
    plt.subplot(1, 2, 1)
    plt.bar(centers, valuefrac, align='center', width = width)
    plt.xscale('log')
    # plt.bar(centers, valuefrac, align='center', width = width)
    plt.xlabel('$d$ [d, in $\mu m$]')
    plt.ylabel(ylab1)

    plt.subplot(1, 2, 2)
    plt.plot(base,percfiner, 'b')
    plt.xscale('log')
    plt.xlabel('$d$ [$\mu m$]')
    plt.ylabel(ylab2)
    distype='PSD num.: '+dist+'\nbin type: '+binmethod
    plt.annotate(distype, xy=(1, 1), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points')

    sizes='Size Stats [$\mu m$]\n$d_{16}=%.0f$\n$d_{50}=%.0f$\n$d_{84}=%.0f$'%(d16, d50, d84)
    plt.annotate(sizes, xy=(1, 0.85), xytext=(12, -12), va='top',xycoords='axes fraction', textcoords='offset points')
    plt.tight_layout()
    plt.savefig(super_path+'/PSD_num'+dist+'_bintype_'+binmethod+'.pdf')

    data = pd.DataFrame({'d_mu':centers, data1:values, 'fraction':valuefrac})
    data.to_csv(super_path+'/PSD_num'+dist+'_bintype_'+binmethod+'.csv',index=False)

    os.chdir(CodePath)
