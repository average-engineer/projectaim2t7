
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq # For FFT
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import decomposition

from dataParser import dataParser # importing the function for reading all the csv files
# Importing data filtering function
from dataFilter import dataFilter
# Importing data cutting function
from dataCut import dataCut


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
    acc[i][:,1] = dataFilter(acc[i][:,1],4,5,sampfreqAcc[i])
    acc[i][:,2] = dataFilter(acc[i][:,2],4,5,sampfreqAcc[i])
    acc[i][:,3] = dataFilter(acc[i][:,3],4,5,sampfreqAcc[i])
    
for j in range(0,len(gyr)):
    gyr[j][:,1] = dataFilter(gyr[j][:,1],4,2,sampfreqGyr[j])
    gyr[j][:,2] = dataFilter(gyr[j][:,2],4,2,sampfreqGyr[j])
    gyr[j][:,3] = dataFilter(gyr[j][:,3],4,2,sampfreqGyr[j])
    
    
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




#%% Cutting Data
acc_cut,gyr_cut = dataCut(acc[13][:,1:4],gyr[13][:,1:4],sampfreqAcc[13],sampfreqGyr[13])

# acc_cut and gyr_cut are 2D arrays containing the X,Y and Z (excluding the time vector) values of the cut data

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

#%% Rotating and Reducing Dimensionality of Data -> Principle Component Analysis
data_gyr = scale(gyr_cut)
pca_gyr = decomposition.PCA(n_components=1) # Number of prinicple component axes = 1
pca_gyr.fit(data_gyr)
trans_gyr = pca_gyr.transform(data_gyr)
trans_df_gyr = pd.DataFrame(trans_gyr)
data_acc = scale(acc_cut)
pca_acc = decomposition.PCA(n_components=1)
pca_acc.fit(data_acc)
trans_acc = pca_acc.transform(data_acc)
trans_df_acc = pd.DataFrame(trans_acc)


acc_rms = np.empty((np.shape(acc_cut)[0]))
for i in range(0,np.shape(acc_cut)[0]):
    acc_rms[i] = np.sqrt((np.power(acc_cut[i,0],2) + np.power(acc_cut[i,1],2) + np.power(acc_cut[i,2],2))/(3))
    
fig = plt.figure(figsize = (10,10))
plt.plot(trans_acc,label = 'PCATime Series')
# plt.plot(acc_rms, label = 'RMS Time Series')
plt.legend()
# plt.show()