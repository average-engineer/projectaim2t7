# Modules
import numpy as np
# import matplotlib.pyplot as plt
from scipy import signal

#%% Function For cutting the data and extracting Motion Sequence

#%% Primarliy the data aat the beginning of the trials have to be left out since
# there may have been some lag in starting the movement at the start of the trials
# and thus the starting data won't be a part of the motion sequence
# Similarly with the end few datapoints when the subject is finishing the trials


# Data is cut based on the occurance of peaks in the data
# The norms of the data are taken along the rows i.e.:
    # Sample Acceleration Data as 2D array:
        #accX   accY    accZ
        #acc_abs(for one row) = square summation of accX, accY and accZ
    # By usig norms, we get one time series with the value of each element in
    # the series being the norm across the rows (X,Y and Z)
        
#%% Function Definition

# Empty placeholder lists
accEmp = []
gyrEmp = []

def dataCut(acc, gyr, accfreq, gyrfreq): # 2D arrays containing data in all 3D coordinates are input to te function
    # Taking norms across columns in each row and converting into one time series
    acc_abs = np.linalg.norm(acc,axis = 1)
    # fig = plt.figure(figsize = (10,10))
    # plt.plot(acc_abs)
    
    # The peaks having having height lesser than 11 m/s2 are not considered
    peaks, _ = signal.find_peaks(acc_abs,height=11) # peaks will be the index numbers at which the peaks (local maxima) occur
    diff_peaks = np.diff(peaks) # diff_peaks gives us the diffference between the index numbers of consecutive peaks
    gap1  = np.argmax(diff_peaks[:20]) # Largest difference index number betwween consecutive peak indices in the first 20 peaks
    gap2  = np.argmax(diff_peaks[-10:]) # Largest difference index number betwween consecutive peak indices in the last 10 peaks
    gap2  = int(gap2 + np.shape(diff_peaks)[0]-10) # gap2 index number is defined wrt start of the indices 

    if (gap2 + np.shape(diff_peaks)[0]) < 11: # For some datasets, the gap2 index becomes negative
        return accEmp,gyrEmp
    
    else:
        # Extracted Motion Sequence
        acc_cut = acc[peaks[gap1+1]:peaks[gap2],:]
        gyr_cut = gyr[peaks[gap1+1]:peaks[gap2],:]
    
        print('Motion Sequence Extracted')
    
        return acc_cut, gyr_cut # Cut 2D arrays returned

    
    