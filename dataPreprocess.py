
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np

from csvread import csvread # importing the function for reading all the csv files
# Importing data filtering function
from dataFilter import dataFilter

## Obtaining all pertinent data for the mentioned subject numbers
accX,accY,accZ,gyrX,gyrY,gyrZ,timeVector,sampFreq = csvread()


# Plotting the Raw Data
fig, (axis1,axis2) = plt.subplots(nrows = 2, ncols = 1, figsize = (8.5,8.5))
# Sample Acceleration
ax = accX[0]
ay = accY[0]
t = timeVector[0]
axis1.plot(t,ax,linewidth = 2)
axis2.plot(t,ay,linewidth = 2)
axis1.set_xlabel('Time (s)')
axis1.set_ylabel('Acceleration in X (m/s2)')
axis2.set_xlabel('Time (s)')
axis2.set_ylabel('Acceleration in Y (m/s2)')

# Plotting the Filtered Data
axFilt = dataFilter(ax,4,10,sampFreq[0])
ayFilt = dataFilter(ay,4,10,sampFreq[0])