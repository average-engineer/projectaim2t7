
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq # For FFT

from dataParser import dataParser # importing the function for reading all the csv files
# Importing data filtering function
from dataFilter import dataFilter


#%% Obtaining all pertinent data for the mentioned subject numbers
AccData,GyrData,experiments = dataParser()

# All sampling Frequencies
sampfreqAcc = list(AccData.keys()) # All acceleration files sampling frequencies
sampfreqGyr = list(GyrData.keys()) # All gyroscope files sampling frequencies

# All csv data
acc = list(AccData.values())
gyr = list(GyrData.values())



# Raw Data
ax = acc[1][:,1]
ay = acc[1][:,2]
az = acc[1][:,3]
ta = acc[1][:,0]

gx = gyr[1][:,1]
gy = gyr[1][:,2]
gz = gyr[1][:,3]
tg = gyr[1][:,0]
print('Raw Data Extracted')

# Filtered Data
# Deciding cut-off frequency for filtering
# Using FFT (Fast Fourier Transform)
# The frequency for which the frequency strength stops having significant peaks, that can 
# be assumed to be the cut-off frequency

yfx = rfft(ax) # rfft(Raw Signal)
xfx = rfftfreq(np.size(ax),1/sampfreqAcc[1]) # rfftfreq(#Datapoints, 1/sampling frequency)
fig = plt.figure(figsize = (10,10))
plt.plot(xfx,abs(yfx))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency Strength')
plt.grid()
# plt.axis([0,50,0,14000])

yfy = rfft(ay) # rfft(Raw Signal)
xfy = rfftfreq(np.size(ay),1/sampfreqAcc[1]) # rfftfreq(#Datapoints, 1/sampling frequency)
fig = plt.figure(figsize = (10,10))
plt.plot(xfy,abs(yfy))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency Strength')
plt.grid()
# plt.axis([0,50,0,14000])

yfz = rfft(az) # rfft(Raw Signal)
xfz = rfftfreq(np.size(az),1/sampfreqAcc[1]) # rfftfreq(#Datapoints, 1/sampling frequency)
fig = plt.figure(figsize = (10,10))
plt.plot(xfz,abs(yfz))
plt.xlabel('Frequency (Hz)')
plt.ylabel('Frequency Strength')
plt.grid()
# plt.axis([0,50,0,14000])

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



axFilt = dataFilter(ax,4,10,sampfreqAcc[0])
ayFilt = dataFilter(ay,4,10,sampfreqAcc[0])
print('Data Filtered')

# Plotting Raw and Filtered Acceleration X
fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8.5,8.5))
axis1.plot(t,ax,linewidth = 2)
axis2.plot(t,axFilt,linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Acceleration in X (m/s2)')
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Filtered Acceleration in X (m/s2)')
axis1.axis([0,30,-15,15])
axis2.axis([0,30,-15,15])
plt.grid()
print('Raw and Filtered Acceleration X Plotted')


# Plotting Raw and Filtered Acceleration Y
fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8.5,8.5))
axis1.plot(t,ay,linewidth = 2)
axis2.plot(t,ayFilt,linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Acceleration in Y (m/s2)')
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Filtered Acceleration in Y (m/s2)')
axis1.axis([0,30,-30,10])
axis2.axis([0,30,-30,10])
print('Raw and Filtered Acceleration Y Plotted')    


