
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np

from dataParser import dataParser # importing the function for reading all the csv files
# Importing data filtering function
from dataFilter import dataFilter


#%% Obtaining all pertinent data for the mentioned subject numbers
AccData,GyrData = dataParser()

# All sampling Frequencies
sampfreqAcc = list(AccData.keys()) # All acceelration files sampling frequencies
sampfreqGyr = list(GyrData.keys()) # All gyroscope files sampling frequencies

# All csv data
acc = list(AccData.values())
gyr = list(GyrData.values())



# Raw Data
ax = acc[0][:,1]
ay = acc[0][:,2]
t = acc[0][:,0]
print('Raw Data Extracted')
# Filtered Data
axFilt = dataFilter(ax,4,5,sampfreqAcc[0])
ayFilt = dataFilter(ay,4,5,sampfreqAcc[0])
print('Data Filtered')

# Plotting Raw and Filtered Acceleration X
fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8.5,8.5))
axis1.plot(t,ax,linewidth = 2)
axis2.plot(t,axFilt,linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Acceleration in X (m/s2)')
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Filtered Acceleration in X (m/s2)')
print('Raw and Filtered Acceleration X Plotted')


# Plotting Raw and Filtered Acceleration Y
fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8.5,8.5))
axis1.plot(t,ay,linewidth = 2)
axis2.plot(t,ayFilt,linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Acceleration in Y (m/s2)')
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Filtered Acceleration in Y (m/s2)')
print('Raw and Filtered Acceleration Y Plotted')


