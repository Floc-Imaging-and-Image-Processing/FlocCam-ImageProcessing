#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Updated 2024-03-22

@author: Kyle Strom
"""
#............................................................................
# user inputs
#............................................................................

# --- directory to process ---

where = '0_Paths.csv'

# --- image values ---

img_sz_x = 4000 # image size x
img_sz_y = 3000 # image size y
# img_sz_x = 1920 # image size x
# img_sz_y = 1080 # image size y

muperpix = 0.925 # muperpix is the number of microns per per pixel (FlocARAZI)
# muperpix = 1.28 # muperpix is the number of microns per per pixel (Lab cam)

# --- filter criteria ---

# minimums
minarea = 9 # min area for a particle in pixels
edge_thickness = 1 # distance from the image edge a particle must exceed to be counted

# focus criteria
focus = 100 # value of max grayscale in edge detection flocs < focus are treated as out of focus

# streaking criteria (filter images impacted by high-speed movement during image capture)
rs = 1 # set to 1 if you want to filter out streaked images
streak_model = './models/streak_remove_1.pickle'

# --- size distribution variables ---

minsize = 10   # min particle size in micron
maxsize = 2000  # max particle size in micron

includestreaks = 0 # set to zero to process sizes without the streak-impacted images

darea = 1 # use 1 to base particle size on area, 0 to base it on the minor axis of the fit ellipse
vdist = 1 # use particle vol for distributions rather than the frequency weighting... if 1, then the w value below does not matter, if 0, w value is used
w = 3 # the distribution weighting value 0 = by number, 1 = by diameter (Ali's), 2 = by area, 3 = by vol
nb = 30 # number of bins used to develop the particle psd and size stats

#............................................................................

# imports

import numpy as np
import glob
import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
import time
import pickle

time1 = time.time()

# find the data directories and setup an output direcectory

datafiletype = '.txt'
output_folder = '0_analysis_output' # make sure this directory does not exist (created directory will be main path/output_folder)

name_list=['Number', 'ImgNo', 'NoInTot', 'Area', 'MeanGreyValue', 'StdDev', 'MinGreyValue', 'MaxGreyValue', 'Perimeter', 'BX', 'BY', 'Width', 'Height', 'Major', 'Minor','Angle', 'Circularity', 'AR','Round', 'Solidity'] # works in the processing in imageJ_code_diffXX.txt

loaded_model = pickle.load(open(streak_model, 'rb'))

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
    
    outputfiles_ns = []
    outputfiles_ys = []

    for k in np.arange(len(sorted_dirs)): #loop over directories
        print('Roaming through folder', int(k+1), 'of', len(sorted_dirs), 'to find you the best flocs :)')
        datafiles = sorted(glob.glob(sorted_dirs[k]+'/*'+'.txt'))
        # print(datafiles)
        # datafiles = sorted(glob.glob(sorted_dirs[0]+'/*'+'.txt'))
        numfiles = len(datafiles)

        counter = 0
        total_particles = 0
        streak_num = 0
        
        image = np.array([])
        image_angle_std = np.array([])
        image_major_minor = np.array([])
        streak_no_streak = np.array([])
                
        for f in np.arange(numfiles): #loop over files in each directory
            datafile=datafiles[f]

            if os.path.exists(datafile):
                # read in the individual particle size file
                temp = pd.read_csv(datafile,  delim_whitespace=True)
                
                # add in the particle number, image number, and particle number in total 
                temp.insert(0, "Number", (np.arange(len(temp))+1).tolist(), True)
                temp.insert(1, "Img_no", ((f+1)*np.ones(len(temp))).astype(int).tolist(), True)
                col_2 = (np.arange(total_particles, len(temp)+ total_particles)+1).tolist()
                temp.insert(2, "No_in_tot",col_2, True)

                # rename the column headers of the master dataframe
                temp.columns.values[:] = name_list[:]
                
                # keep track of the total number of particles (before filtering)
                total_particles = total_particles + len(temp)
                
                ## Define critria to drop rows (flocs) and filter out bad points

                totalparticles = len(temp)
                temp = temp[(temp['MaxGreyValue']>=focus) & # clarity crit
                (temp['BX']+temp['Width']<img_sz_x-edge_thickness) & (temp['BX']>edge_thickness) &  # edge crit
                (temp['BY']+temp['Height']<img_sz_y-edge_thickness) & (temp['BY']>edge_thickness) & # edge crit
                (temp['Area']>=minarea) ] # min particle size in square pixels
                
                angle_std = np.std(temp.Angle)
                axisR = np.mean(temp.Major/temp.Minor)
                frac_num_focus = len(temp)/total_particles
                area_p = np.mean(temp['Area']/temp['Perimeter'])
                
                input = np.array([[angle_std,axisR,frac_num_focus,area_p]])
                
                if np.isnan(input).any() == False:
                    yes_streak = loaded_model.predict(input)[0]
                else:
                    yes_streak = 1
                    
                if yes_streak == 1:
                    streak_num = streak_num + 1
                
                image = np.append(image,f)
                image_angle_std = np.append(image_angle_std,angle_std)
                image_major_minor = np.append(image_major_minor,axisR)
                streak_no_streak = np.append(streak_no_streak,yes_streak)
                
                temp['Streak_impact'] = yes_streak
                
                # if angle_std < angle_std_cr and axisR > axisR_cr:
                #     temp_f['Streak_impact'] = 1
                #     # streak_num = streak_num + 1
                #     # print('streak image =', f)
                #     streak_image = np.append(streak_image,f)
                #     streak_angle_std = np.append(streak_angle_std,angle_std)
                #     streak_major_minor = np.append(streak_major_minor,axisR)
                # else:
                #     temp_F['Streak_impact'] = 0

                if counter==0:
                    frames0 = temp
                else:
                    frames0 = pd.concat([frames0,temp])
                counter = counter + 1

        # print and save information on number of images impacted by streaks and whether the size stats data include those images
        
        streak_df = pd.DataFrame({'Image': image, 'Avg_Angle_StD': image_angle_std, 'Avg_Major_Minor':image_major_minor, 'Streak':streak_no_streak})
        streak_df.to_csv(super_path+'/Streak-impacted_images.csv',index=False)
        
        print('Number of total images =', counter, ', Number of images impacted by streaking =',streak_num)
        
        f = open(super_path+ '/Readme.txt', 'x')
        f.write('Processing Information --- \n')
        f.write('- Number of total images processed = '+str(counter)+'\n')
        f.write('- Number of images impacted by streaking = '+str(streak_num)+'\n')
        if includestreaks == 0:
            f.write('- Do size statistics include potential streak-impacted images? NO')
        else:
            f.write('- Do size statistics include potential streak-impacted images? YES')
        f.close()

        # filter out all particles associated with a streak-impacted image
        
        no_streaks = frames0[(frames0['Streak_impact']==0)]
        
        # save the data 

        dest_file_csv = super_path+ '/'+str(sorted_dirs[k])+'.csv'
        dest_file_csv_all = super_path+ '/'+str(sorted_dirs[k])+'_all.csv'

        frames0.to_csv(dest_file_csv_all,index=False)
        no_streaks.to_csv(dest_file_csv,index=False)
        
        outputfiles_ns.append(dest_file_csv)
        outputfiles_ys.append(dest_file_csv_all)
    
    # pull in size data and start processing --------------------------------

    # find the nubmer of files to process
    # gsdfiles_pixel = sorted(glob.glob(super_path+'/'+"*.csv"))
    if includestreaks == 0:
        gsdfiles_pixel = outputfiles_ns
    else:
        gsdfiles_pixel = outputfiles_ys
        
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
