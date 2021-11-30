
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq # For FFT
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder



from dataParser import dataParser # importing the function for reading all the csv files
# Importing data filtering function
from dataFilter import dataFilter
# Importing data cutting function
from dataCut import dataCut
# Importing data segmenting function
from dataSegment import dataSegment

#%% Global Variables
global coFreqAcc # Cut-off frequency for acceleration data
global coFreqGyr # Cut-off frequency for gyroscope data
coFreqAcc = 5 # Hz
coFreqGyr = 1 # Hz

#%% Method for Rotating and Standardising Data
# 1). PCA (principle component analysis) -> 'PCA'
# 2). RMS of normalised time series -> 'RMS'

rot = 'RMS'

#%% Dataframes for accumulating data
colNames = ['Gait']
accData = pd.DataFrame(columns = colNames)
gyrData = pd.DataFrame(columns = colNames)
# =============================================================================
# accData = pd.DataFrame(columns = colNames)
# gyrData = pd.DataFrame(columns = colNames)
# =============================================================================


#%% Obtaining all pertinent data for the mentioned subject numbers
AccData,GyrData,experiments = dataParser()

# All sampling Frequencies
sampfreqAcc = list(AccData.keys()) # All acceleration files sampling frequencies
sampfreqGyr = list(GyrData.keys()) # All gyroscope files sampling frequencies

# All csv data
acc = list(AccData.values())
gyr = list(GyrData.values())



#%% Extracting Sample Raw Data
ax = acc[1][:,1]
ay = acc[1][:,2]
az = acc[1][:,3]
ta = acc[1][:,0]

gx = gyr[1][:,1]
gy = gyr[1][:,2]
gz = gyr[1][:,3]
tg = gyr[1][:,0]
print('Raw Data Extracted')

#%% Plotting Raw Data
fig,(axis1,axis2,axis3) = plt.subplots(nrows = 3, ncols = 1, figsize = (8,8))

axis1.plot(acc[0][:,0],acc[0][:,1],linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Raw Acceleration in X (m/s2)')
axis1.axis([0,30,-15,15])
axis1.grid()


axis2.plot(acc[0][:,0],acc[0][:,2],linewidth = 2)
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Raw Acceleration in Y (m/s2)')
axis2.axis([0,30,-20,5])
axis2.grid()

axis3.plot(acc[0][:,0],acc[0][:,3],linewidth = 2)
axis3.set_xlabel('Time (s)')
axis3.set_ylabel('Raw Acceleration in Z (m/s2)')
axis3.axis([0,30,-15,15])
axis3.axis([0,30,-15,15])
axis3.grid()
print('Raw Acceleration Plotted')

fig,(axis1,axis2,axis3) = plt.subplots(nrows = 3, ncols = 1, figsize = (8,8))

axis1.plot(gyr[0][:,0],gyr[0][:,1],linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Raw Angular Velocity in X (rad/s2)')
axis1.axis([0,30,-15,15])
axis1.grid()


axis2.plot(gyr[0][:,0],gyr[0][:,2],linewidth = 2)
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Raw Angular Velocity in Y (rad/s2)')
axis2.axis([0,30,-15,15])
axis2.axis([0,30,-15,15])
axis2.grid()

axis3.plot(gyr[0][:,0],gyr[0][:,3],linewidth = 2)
axis3.set_xlabel('Time (s)')
axis3.set_ylabel('Raw Angular Velocity in Z (rad/s2)')
axis3.axis([0,30,-15,15])
axis3.axis([0,30,-15,15])
axis3.grid()
print('Raw Angular Velocity Plotted')

#%% Using FFT to get an idea about the cut-off frequencies
# Deciding cut-off frequency for filtering
# Using FFT (Fast Fourier Transform)
# The frequency for which the frequency strength stops having significant peaks, that can 
# be assumed to be the cut-off frequency

# =============================================================================
# yfx = rfft(ax) # rfft(Raw Signal)
# xfx = rfftfreq(np.size(ax),1/sampfreqAcc[1]) # rfftfreq(#Datapoints, 1/sampling frequency)
# fig = plt.figure(figsize = (10,10))
# plt.plot(xfx,abs(yfx))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Frequency Strength')
# plt.grid()
# # plt.axis([0,50,0,14000])
# 
# yfy = rfft(ay) # rfft(Raw Signal)
# xfy = rfftfreq(np.size(ay),1/sampfreqAcc[1]) # rfftfreq(#Datapoints, 1/sampling frequency)
# fig = plt.figure(figsize = (10,10))
# plt.plot(xfy,abs(yfy))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Frequency Strength')
# plt.grid()
# # plt.axis([0,50,0,14000])
# 
# yfz = rfft(az) # rfft(Raw Signal)
# xfz = rfftfreq(np.size(az),1/sampfreqAcc[1]) # rfftfreq(#Datapoints, 1/sampling frequency)
# fig = plt.figure(figsize = (10,10))
# plt.plot(xfz,abs(yfz))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Frequency Strength')
# plt.grid()
# # plt.axis([0,50,0,14000])
# =============================================================================

# =============================================================================
# yfxg = rfft(gx) # rfft(Raw Signal)
# xfxg = rfftfreq(np.size(gx),1/sampfreqGyr[10]) # rfftfreq(#Datapoints, 1/sampling frequency)
# fig = plt.figure(figsize = (10,10))
# plt.plot(xfxg,abs(yfxg))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Frequency Strength')
# plt.grid()
# 
# yfyg = rfft(gy) # rfft(Raw Signal)
# xfyg = rfftfreq(np.size(gy),1/sampfreqGyr[10]) # rfftfreq(#Datapoints, 1/sampling frequency)
# fig = plt.figure(figsize = (10,10))
# plt.plot(xfyg,abs(yfyg))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Frequency Strength')
# plt.grid()
# 
# yfzg = rfft(gz) # rfft(Raw Signal)
# xfzg = rfftfreq(np.size(gz),1/sampfreqGyr[10]) # rfftfreq(#Datapoints, 1/sampling frequency)
# fig = plt.figure(figsize = (10,10))
# plt.plot(xfzg,abs(yfzg))
# plt.xlabel('Frequency (Hz)')
# plt.ylabel('Frequency Strength')
# plt.grid()
# =============================================================================




#%% Filtering the Data
for i in range(0,len(acc)):
    acc[i][:,1] = dataFilter(acc[i][:,1],4,coFreqAcc,sampfreqAcc[i])
    acc[i][:,2] = dataFilter(acc[i][:,2],4,coFreqAcc,sampfreqAcc[i])
    acc[i][:,3] = dataFilter(acc[i][:,3],4,coFreqAcc,sampfreqAcc[i])
    
for j in range(0,len(gyr)):
    gyr[j][:,1] = dataFilter(gyr[j][:,1],4,coFreqGyr,sampfreqGyr[j])
    gyr[j][:,2] = dataFilter(gyr[j][:,2],4,coFreqGyr,sampfreqGyr[j])
    gyr[j][:,3] = dataFilter(gyr[j][:,3],4,coFreqGyr,sampfreqGyr[j])
    
    
print('Data Filtered')

#%% Plotting Filtered Data
fig,(axis1,axis2,axis3) = plt.subplots(nrows = 3, ncols = 1, figsize = (8,8))

axis1.plot(acc[0][:,0],acc[0][:,1],linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Filtered Acceleration in X (m/s2)')
axis1.axis([0,30,-15,15])
axis1.grid()

axis2.plot(acc[0][:,0],acc[0][:,2],linewidth = 2)
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Filtered Acceleration in Y (m/s2)')
axis2.axis([0,30,-15,15])
axis2.axis([0,30,-15,15])
axis2.grid()

axis3.plot(acc[0][:,0],acc[0][:,3],linewidth = 2)
axis3.set_xlabel('Time (s)')
axis3.set_ylabel('Filtered Acceleration in Z (m/s2)')
axis3.axis([0,30,-15,15])
axis3.axis([0,30,-15,15])
axis3.grid()
print('Filtered Acceleration Plotted')


fig,(axis1,axis2,axis3) = plt.subplots(nrows = 3, ncols = 1, figsize = (8,8))

axis1.plot(gyr[0][:,0],gyr[0][:,1],linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Filtered Angular Velocity in X (rad/s2)')
axis1.axis([0,30,-5,5])
axis1.grid()


axis2.plot(gyr[0][:,0],gyr[0][:,2],linewidth = 2)
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Filtered Angular Velocity in Y (rad/s2)')
axis2.axis([0,30,-15,15])
axis2.axis([0,30,-5,5])
axis2.grid()

axis3.plot(gyr[0][:,0],gyr[0][:,3],linewidth = 2)
axis3.set_xlabel('Time (s)')
axis3.set_ylabel('Filtered Angular Velocity in Z (rad/s2)')
axis3.axis([0,30,-5,5])
axis3.grid()
print('Filtered Angular Velocity Plotted')  

#%% Further Preprocessing
# Empty List for New Processed acceleration data
acc1 = []
# Empty List for New Processed gyroscope data
gyr1 = []

count = 0
firstsegAcc = []
firstsegGyr = []
for i in range(0,len(acc)):
    #%% Cutting Data
    acc_cut,gyr_cut = dataCut(acc[i][:,1:4],gyr[i][:,1:4],sampfreqAcc[i],sampfreqGyr[i])
    
    if len(acc_cut) == 0 or len(gyr_cut) == 0:
        continue
    
    if np.shape(gyr_cut)[0] == 0 or np.shape(acc_cut)[0] == 0: # Some issues in some trials where gyroscope data vanishes
                                  # after cutting the data -> Trials where that is happening 
                                  # are omitted for now
        continue
    
    #%% Rotating and Reducing Dimensionality of Data -> Principle Component Analysis
    
    if rot == 'PCA':
        
        data_gyr = scale(gyr_cut)
        pca_gyr = decomposition.PCA(n_components=1) # Number of prinicple component axes = 1
        pca_gyr.fit(data_gyr)
        trans_gyr = pca_gyr.transform(data_gyr)
        # trans_df_gyr = pd.DataFrame(trans_gyr)
        data_acc = scale(acc_cut)
        pca_acc = decomposition.PCA(n_components=1)
        pca_acc.fit(data_acc)
        trans_acc = pca_acc.transform(data_acc)
        # trans_df_acc = pd.DataFrame(trans_acc)
        
        
    # Rotating and Reducing Dimensionality of Data -> RMS Time series    
        
    elif rot == 'RMS':
    
        # Normalising the data
        
        # Acceleration
        # Means of X,Y,Z time series
        x_mean = np.mean(acc_cut[:,0])
        y_mean = np.mean(acc_cut[:,1])
        z_mean = np.mean(acc_cut[:,2])
        
        # Max of X,Y,Z time series
        x_max = np.max(acc_cut[:,0])
        y_max = np.max(acc_cut[:,1])
        z_max = np.max(acc_cut[:,2])
        
        trans_acc = np.empty((np.shape(acc_cut)[0],1)) # empty array for storing the RMS time series of acceleration data
        
        # Normalised Data -> Between -1 and 1
        # Values smaller than mean will be in [-1,0] and values greater than mean will be [0,1]
        for ii in range(0,np.shape(acc_cut)[0]):
    
            x_norm = (acc_cut[ii,0] - x_mean)/(x_max - x_mean)
            y_norm = (acc_cut[ii,0] - y_mean)/(y_max - y_mean)
            z_norm = (acc_cut[ii,0] - z_mean)/(z_max - z_mean)
            
            trans_acc[ii] = np.sqrt((np.power(x_norm,2) + np.power(y_norm,2) + np.power(z_norm,2))/(3))
            
            
        # Gyroscope
        # Means of X,Y,Z time series
        x_mean = np.mean(gyr_cut[:,0])
        y_mean = np.mean(gyr_cut[:,1])
        z_mean = np.mean(gyr_cut[:,2])
        
        # Max of X,Y,Z time series
        x_max = np.max(gyr_cut[:,0])
        y_max = np.max(gyr_cut[:,1])
        z_max = np.max(gyr_cut[:,2])
        
        trans_gyr = np.empty((np.shape(gyr_cut)[0],1)) # empty array for storing the RMS time series of acceleration data
        
        # Normalised Data -> Between -1 and 1
        # Values smaller than mean will be in [-1,0] and values greater than mean will be [0,1]
        for ii in range(0,np.shape(gyr_cut)[0]):
    
            x_norm = (gyr_cut[ii,0] - x_mean)/(x_max - x_mean)
            y_norm = (gyr_cut[ii,0] - y_mean)/(y_max - y_mean)
            z_norm = (gyr_cut[ii,0] - z_mean)/(z_max - z_mean)
            
            trans_gyr[ii] = np.sqrt((np.power(x_norm,2) + np.power(y_norm,2) + np.power(z_norm,2))/(3))
            
        
        
            
    
    #%% Segmenting the data and resampling the cycles
    segAcc, segGyr = dataSegment(trans_acc[:,0],trans_gyr[:,0],rot)

    
    if i == 6:
        firstsegAcc = segAcc
        firstsegGyr = segGyr

    
    
    #%% Accumilating the Data
# =============================================================================
#     accData['Trials'] = accData['Trials'].astype('object')
#     gyrData['Trials'] = gyrData['Trials'].astype('object')
# =============================================================================
    
    
    #%% Removing Malicious Data
    
    # Array for storing mean of each cycle
    meanCycle = np.zeros(len(segAcc))
    for ii in range(0,len(segAcc)):
        meanCycle[ii] = np.mean(segAcc[ii])
        
    meanCycleMean = np.mean(meanCycle) # Mean of all cycle means
    
    # Standard Deviation of cycle means wrt mean of cycle means
    sigma = np.std(meanCycle)
    
    segAcc_nonmal =[]
    
    for ii in range(0,meanCycle.shape[0]):
        if meanCycle[ii] >= (meanCycleMean + sigma) or meanCycle[ii] <= (meanCycleMean - sigma):
            continue # The particular cycle is removed
        else:
            segAcc_nonmal.append(segAcc[ii]) # List containing non-malicious cycles as arrays
            
    
    # Array for storing mean of each cycle
    meanCycle = np.zeros(len(segGyr))
    for iter in range(len(segGyr)):
        meanCycle[iter] = np.mean(segGyr[iter])
        
    meanCycleMean = np.mean(meanCycle) # Mean of all cycle means
    
    # Standard Deviation of cycle means wrt mean of cycle means
    sigma = np.std(meanCycle)
    
    segGyr_nonmal = []
    
    for ii in range(len(segGyr)):
        if meanCycle[ii] >= (meanCycleMean + sigma) or meanCycle[ii] <= (meanCycleMean - sigma)  :
            continue # The particular cycle is removed
        else:
            segGyr_nonmal.append(segGyr[ii]) # List containing non-malicious cycles as arrays
    
    
# =============================================================================
#     accData.at[count,'Trials'] = segAcc_nonmal
#     
#     gyrData.at[count,'Trials'] = segGyr_nonmal
# =============================================================================
    
    cnt = 0
    for ii in range(0,len(segAcc_nonmal)):
        for jj in range(0,segAcc_nonmal[ii].shape[0]):
            accData.at[count,cnt] = segAcc_nonmal[ii][jj]
            cnt = cnt + 1
            
    cnt = 0
    for ii in range(0,len(segGyr_nonmal)):
        for jj in range(0,segGyr_nonmal[ii].shape[0]):
            gyrData.at[count,cnt] = segGyr_nonmal[ii][jj]
            cnt = cnt + 1
    
# =============================================================================
#     accData.at[count,'Sensor'] = 'Accelerometer'
#     gyrData.at[count,'Sensor'] = 'Gyroscope'
# =============================================================================
    
    accData.at[count,'Gait'] = experiments[i]
    gyrData.at[count,'Gait'] = experiments[i]

    # Data stores alternatively as lists
    acc1.append(segAcc_nonmal)
    gyr1.append(segGyr_nonmal)
    
    count = count+ 1
    
        
#%% Sample Cycles

# Acceleration
# =============================================================================
# fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8,8))
# for i in range(len(accData.at[2,'Trials'])):
#     axis1.plot(accData.at[2,'Trials'][i])
# axis1.title.set_text('Malicious Data Excluded')
# axis1.grid()
# 
# for i in range(len(firstsegAcc)):
#     axis2.plot(firstsegAcc[i])
# axis2.title.set_text('Malicious Data Included')
# axis2.grid()
# plt.show()
# 
# 
# # Gyroscope
# fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8,8))
# for i in range(len(gyrData.at[2,'Trials'])):
#     axis1.plot(gyrData.at[2,'Trials'][i])
# axis1.title.set_text('Malicious Data Excluded')
# axis1.grid()
# 
# for i in range(len(firstsegGyr)):
#     axis2.plot(firstsegGyr[i])
# axis2.title.set_text('Malicious Data Included')
# axis2.grid()
# plt.show()
# =============================================================================
#%% acc_cut and gyr_cut are 2D arrays containing the X,Y and Z (excluding the time vector) values of the cut data

# =============================================================================
# fig,(axis1,axis2,axis3) = plt.subplots(nrows = 3, ncols = 1, figsize = (7,7))
# axis1.plot(acc_cut[:,0])
# axis1.grid()
# 
# axis2.plot(acc_cut[:,1])
# axis2.grid()
# 
# axis3.plot(acc_cut[:,2])
# axis3.grid()
# =============================================================================


#%%
    
# =============================================================================
# fig = plt.figure(figsize = (10,10))
# plt.plot(trans_acc,label = 'PCATime Series')
# # plt.plot(acc_rms, label = 'RMS Time Series')
# plt.legend()
# # plt.show()
# =============================================================================




#%% Setting Labels
label = LabelEncoder()
accData['label'] = label.fit_transform(accData['Gait'])
accData.head()




