#%% Function for segmenting the data

from scipy import signal
import numpy as np

#%% Function Definition
def dataSegment(acc,gyr,rot):
    
    if rot == 'PCA':
        # Peak Finding when PCA is used
        peaksAcc, _ = signal.find_peaks(acc,height = 2) # Returns indices of peaks in acceleration data
        peaksGyr, _ = signal.find_peaks(gyr,height = 1.4) # Returns indices of peaks in gyroscope data
        
    elif rot == 'RMS':
        # Peak Finding when RMS of normalised values are used
        peaksAcc, _ = signal.find_peaks(acc,height = 1.2) # Returns indices of peaks in acceleration data
        peaksGyr, _ = signal.find_peaks(gyr,height = 0.4) # Returns indices of peaks in gyroscope data
    
# ===============================s==============================================
#     diff_peaksAcc = np.diff(peaksAcc)
#     diff_peaksGyr = np.diff(peaksGyr)
# =============================================================================
    
    # We return a list wherer each element of the list is the individual cycle/sample
    # The data arrays are sliced into multiple cycles on the basis of peak indices obtained
    
    # Initialising lists
    segAcc = []
    segGyr = []

# =============================================================================
#     segAcc = np.empty((peaksAcc.shape[0],100))
#     segGyr = np.empty((peaksGyr.shape[0],100))
#     
# =============================================================================
    jj = 0
    for ii in peaksAcc:
        cycAcc = acc[jj:ii + 1] # Cycle obtained as array
        cycAcc = signal.resample(cycAcc,100)
        segAcc.append(cycAcc)
        jj = ii + 1
        #print(segAcc,type(segAcc))
            
        
        
    jj = 0
    for ii in peaksGyr:
        cycGyr = gyr[jj:ii + 1] # Cycle obtained 
        cycGyr = signal.resample(cycGyr,100)
        segGyr.append(cycGyr)
        jj = ii + 1

    return segAcc,segGyr # Lists with each elements being the segmented cycles are returned
