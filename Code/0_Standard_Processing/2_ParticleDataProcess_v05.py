#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May  5 12:51:10 2022

@author: strom-adm
"""

#user inputs
#............................................................................

# directory to process
where = '0_Paths.csv'

# processing values
focus = 100 # value of max grayscale in edge detection flocs < focus are treated as out of focus
minarea = 9 # min area for a particle in pixels
muperpix = 0.925 # muperpix is the number of microns per per pixel (FlocARAZI)
# muperpix = 1.28 # muperpix is the number of microns per per pixel (Lab cam)

darea = 1 # use 1 to base particle size on area, 0 to base it on the minor axis of the fit ellipse

minsize = 10   # min particle size in micron
maxsize = 2000  # max particle size in micron

vdist = 1 # use particle vol for distributions rather than the frequency weighting... if 1, then the w value below does not matter, if 0, w value is used
w = 3 # the distribution weighting value 0 = by number, 1 = by diameter (Ali's), 2 = by area, 3 = by vol
nb = 30 # number of bins used to develop the particle psd and size stats

img_sz_x = 4000 # image size x
img_sz_y = 3000 # image size y
# img_sz_x = 1920 # image size x
# img_sz_y = 1080 # image size y
edge_thickness = 1 # distance from the image edge a particle must exceed to be counted

ms_model = 'sand_mud_models/SVM-linear-2.pickle'

#............................................................................

# imports

import numpy as np
import glob
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import time
time1 = time.time()
import pickle

# load the mud/sand classifier model
loaded_model = pickle.load(open(ms_model, 'rb'))

# find the data directories and setup an output direcectory

datafiletype = '.txt'
output_folder = '0_analysis_output' # make sure this directory does not exist (created directory will be main path/output_folder)

CodePath = os.getcwd()

if where == 'local':
    path_main = os.getcwd()
else:
    paths = pd.read_csv(where)

for j in range(1,len(paths)):
    path_main = paths.iloc[j,1]
    print(path_main)

    # os.chdir(path_main)

    super_path = os.path.join(path_main,output_folder)
    if os.path.isdir(super_path) == True:
        shutil.rmtree(super_path)

    dirs_1 = [f for f in os.listdir(path_main) if os.path.isdir(os.path.join(path_main, f)) if (not(f.startswith('.ipynb_checkpoints')))]
    sorted_dirs = sorted(dirs_1)
    print(sorted_dirs)

    if os.path.isdir(super_path) != True:
        os.mkdir(super_path)

    # create a master combined dataframe that integrates all of the individual image output .txt files - will be used for single PSD

    os.chdir(path_main)

    for k in np.arange(len(sorted_dirs)): #loop over directories
        print('Roaming through folder', int(k+1), 'of', len(sorted_dirs), 'to find you the best flocs :)')
        datafiles = sorted(glob.glob(sorted_dirs[k]+'/*'+'.txt'))
        # print(datafiles)
        # datafiles = sorted(glob.glob(sorted_dirs[0]+'/*'+'.txt'))
        numfiles = len(datafiles)

        counter = 0
        for f in np.arange(numfiles): #loop over files in each directory
            datafile=datafiles[f]

            if os.path.exists(datafile):
                temp = pd.read_csv(datafile,  delim_whitespace=True)
                temp.insert(0, "Number", (np.arange(temp.shape[0])+1).tolist(), True)
                if counter==0:
                    col_2 = (np.arange(temp.shape[0])+1).tolist()
                else:
                    col_2 = (np.arange(frames0.shape[0], temp.shape[0]+ frames0.shape[0])+1).tolist()
                temp.insert(1, "Img_no", ((f+1)*np.ones(temp.shape[0])).astype(int).tolist(), True)
                temp.insert(2, "No_in_tot",col_2, True)

                if counter==0:
                    frames0 = temp
                else:
                    frames0 = pd.concat([frames0,temp])
                counter = counter + 1

        # rename the column headers of the master dataframe

        name_list=['Number', 'ImgNo', 'NoInTot', 'Area', 'MeanGreyValue', 'StdDev', 'MinGreyValue', 'MaxGreyValue',
        'Perimeter', 'BX', 'BY', 'Width', 'Height', 'Major', 'Minor',' Angle', 'Circularity', 'AR',
        'Round', 'Solidity'] # works in the processing in imageJ_code_diffXX.txt

        frames0.columns.values[:] = name_list[:]

        ## Define critria to drop rows (flocs) and filter out bad points

        frames0 = frames0[(frames0['MaxGreyValue']>=focus) & # clarity crit
        (frames0['BX']+frames0['Width']<img_sz_x-edge_thickness) & (frames0['BX']>edge_thickness) &  # edge crit
        (frames0['BY']+frames0['Height']<img_sz_y-edge_thickness) & (frames0['BY']>edge_thickness) & # edge crit
        (frames0['Area']>=minarea) ] # min particle size in square pixels
        
        # classify as mud or sand
        
        inputdata = frames0[['Area','Perimeter','Major','Minor','Circularity','AR','Round','Solidity']].to_numpy()
        
        y_pred = loaded_model.predict(inputdata)
        frames0['sand'] = y_pred
        
        # save the file

        # dest_file_csv = super_path+ '/'    +str(int(k+1)).zfill(3)+'.csv'
        # dest_file_csv = output_folder+ '/'+str(sorted_dirs[k])+'.csv'
        dest_file_csv = super_path+ '/'+str(sorted_dirs[k])+'.csv'

        frames0.to_csv(dest_file_csv,index=False)

    # pull in size data and start processing --------------------------------

    # find the nubmer of files to process
    gsdfiles_pixel = sorted(glob.glob(super_path+'/'+"*.csv"))
    first = 1
    last=len(gsdfiles_pixel)

    # start the dataframe out by reading in the first set of size information
    filename = gsdfiles_pixel[0]
    area_pixel = pd.read_csv(filename,usecols=[3])
    minor_axis = pd.read_csv(filename,usecols=[14])

    # create the larger master dataframe
    for i in np.arange(1,last): #for empty end-test images
        filename=gsdfiles_pixel[i]
        area_i = pd.read_csv(filename,usecols=[3])
        area_pixel = pd.concat([area_pixel,area_i],axis=1)
        minor_i = pd.read_csv(filename,usecols=[14])
        minor_axis = pd.concat([minor_axis,minor_i],axis=1)

    area_pixel.columns = np.arange(first,len(area_pixel.columns)+1) # rename the columns headers to match file names
    minor_axis.columns = np.arange(first,len(minor_axis.columns)+1) # rename the columns headers to match file names

    # input the constants for conversion from pixels to µm

    a_mu2 = area_pixel*muperpix**2 # make a floc area by microns
    minor_mu = minor_axis*muperpix

    if(darea == 1):
        d_mu = np.sqrt(4*a_mu2/np.pi) # calculate the floc spherical diameter in pixels and put everything in microns
    else:
        d_mu = minor_mu.copy() # use the minor axis of the fit ellipse

    # save the large master d_mu file as a .csv

    d_mu_file = super_path+'/'+'d_mu.csv'
    d_mu.to_csv(d_mu_file,index=False)



    # process the whole file to get stats based on specified distribution type (you need to define the type of weighting)

    x=np.linspace(0,9,10)

    d16_mu = np.linspace(0,len(d_mu.columns)-1,len(d_mu.columns))
    d50_mu = np.linspace(0,len(d_mu.columns)-1,len(d_mu.columns))
    d84_mu = np.linspace(0,len(d_mu.columns)-1,len(d_mu.columns))

    first = 0
    last = len(d_mu.columns)

    if vdist == 1:

        Sed_vol_uL=np.linspace(0,len(d_mu.columns)-1,len(d_mu.columns))

        for i in range(first,last):
            d=np.array(d_mu.iloc[:,i])
            d=d[~np.isnan(d)]
            dlog=np.log(d)
            dvol_uL = 1e-9*(np.pi/6)*d**3 # volume in microliters (1 micron^3 = 1e-9 microliters)

            values, base = np.histogram(dlog, bins=nb, weights=dvol_uL)

            cumulative = np.cumsum(values) # adds up the frequencies (total is the total number of points)
            perc = cumulative/cumulative[len(cumulative)-1]
            percfiner = perc
            percfiner = np.insert(percfiner,0,0)
            d16_mu[i-1]=np.exp(np.interp(0.16,percfiner,base))
            d50_mu[i-1]=np.exp(np.interp(0.5,percfiner,base))
            d84_mu[i-1]=np.exp(np.interp(0.84,percfiner,base))
            Sed_vol_uL[i-1]=np.sum(values)

        # create a dataframe and writes the stats to a csv file

        dstatsfile=super_path+'/'+'dstats_by_volume.csv'

        dstats=pd.DataFrame([d16_mu,d50_mu,d84_mu,Sed_vol_uL])
        dstats=dstats.transpose()
        dstats.columns = ['d16_mu','d50_mu','d84_mu','Sed_vol_uL']
        dstats.index = np.arange(1, len(dstats) + 1) # changes the row index so that it starts at 1 instead of 0
        dstats.index.name = 'min'

        dstats.to_csv(dstatsfile)

        #plot the distribution stats
        plt.subplot()
        plt.plot(d16_mu, 's-', label = '$d_{16}$')
        plt.plot(d50_mu, 'o-',label = '$d_{50}$')
        plt.plot(d84_mu, 'v-', label = '$d_{84}$')
        #ylim(0,250)
        plt.xlabel('$t$ [min]')
        plt.ylabel('$d_{f}$ [µm]')
        plt.legend()
        plt.title('by volume')
        plt.tight_layout()
        plt.savefig(super_path+'/SizeStats-timeseries-by-vol.pdf')
        plt.close()

    else:

        for i in np.arange(first,last):
            d = np.array(d_mu.iloc[:,i])
            d = d[~np.isnan(d)]
            dlog = np.log(d)
            values, base = np.histogram(dlog, bins=nb)
            values = values*(np.exp((base[1:]+base[:-1])/2))**w # include this to do the weighting
            cumulative = np.cumsum(values) # adds up the frequencies (total is the total number of points)
            perc = cumulative/cumulative[len(cumulative)-1]
            percfiner = perc
            percfiner = np.insert(percfiner,0,0)
            d16_mu[i-1] = np.exp(np.interp(0.16,percfiner,base))
            d50_mu[i-1] = np.exp(np.interp(0.5,percfiner,base))
            d84_mu[i-1] = np.exp(np.interp(0.84,percfiner,base))

        # create a dataframe and writes the stats to a csv file

        dstats = pd.DataFrame([d16_mu,d50_mu,d84_mu])
        dstats = dstats.transpose()
        dstats.columns = ['d16_mu','d50_mu','d84_mu']
        dstats.index = np.arange(1, len(dstats) + 1) # changes the row index so that it starts at 1 instead of 0
        dstats.index.name = 'min'

        dstatsfile = super_path+'/'+'dstats_by_'+'d'+str(w)+'_weighting.csv'
        dstats.to_csv(dstatsfile)

        #plot the distribution stats time series

        plt.subplot()
        plt.plot(d16_mu, 's-', label = '$d_{16}$')
        plt.plot(d50_mu, 'o-',label = '$d_{50}$')
        plt.plot(d84_mu, 'v-', label = '$d_{84}$')
        plt.xlabel('$t$ [min]')
        plt.ylabel('$d_{f}$ [µm]')
        plt.legend()
        plt.title('d**'+str(w)+' weighting')
        plt.tight_layout()
        plt.savefig(super_path+'/SizeStats-timeseries-d**'+str(w)+'-weighting.pdf')
        plt.close()

    os.chdir(CodePath)
