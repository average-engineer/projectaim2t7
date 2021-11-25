import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import re
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import decomposition

def filter_data(sensordata, fs=200, fc=5):
    sensordata_filt = np.zeros(sensordata.shape)
    
    w = fc / (fs / 2)
    b, a = signal.butter(fc,w, 'low')
    
    sensordata_filt[:,0] = signal.filtfilt(b,a, sensordata[:,0])
    sensordata_filt[:,1] = signal.filtfilt(b,a, sensordata[:,1])
    sensordata_filt[:,2] = signal.filtfilt(b,a, sensordata[:,2])
    
    return sensordata_filt
    
def cut_data(acc_filt, gyr_filt, freq):
    acc_abs = np.linalg.norm(acc_filt,axis=1)
    

    peaks, _ = signal.find_peaks(acc_abs,height=11, distance=freq/2)
    diff_peaks =np.diff(peaks)
    gap1  = np.argmax(diff_peaks[:20])
    gap2  = np.argmax(diff_peaks[-10:])
    gap2  = int(gap2 + np.shape(diff_peaks)-10)

    
    acc_cut = acc_filt[peaks[gap1+1]:peaks[gap2],:]
    gyr_cut = gyr_filt[peaks[gap1+1]:peaks[gap2],:]
    
    return acc_cut, gyr_cut

def seg_data_acc(trans_df_acc):
    peaks, _ = signal.find_peaks(trans_df_acc)
    diff_peaks =np.diff(peaks)

    return peaks,diff_peaks




paths =glob.glob(r'./Data/Smartphone3/*')

for path in paths:
    name = re.split('/',path)
    experiment_name = name[-1]
    splitexp = re.split('_',experiment_name)
    subj=splitexp[0]
    gait=splitexp[1]
    if not re.search(r'red', gait) and re.search(r'216',subj) and re.search(r'downstairs01',gait):
        # re.search(r'215',subj) or re.search(r'217',subj)
        gyr_file = glob.glob(path+'/Gyroscope.csv')
        acc_file = glob.glob(path+'/Accelerometer.csv')
        data_gyr = np.loadtxt(gyr_file[0], delimiter=',', skiprows=1)
        gyr = data_gyr[:,1:4]
        time_gyr = data_gyr[:,0]
        data_acc = np.loadtxt(acc_file[0], delimiter=',', skiprows=1)
        acc = data_acc[:,1:4]
        time_acc = data_acc[:,0]
        samp_freq_gyr = np.round(len(time_gyr)/(time_gyr[-1]))
        sampling_frequency_gyr = np.round(len(time_gyr)/(time_gyr[-1]))
        samp_freq_acc = np.round(len(time_acc)/(time_acc[-1]))
        sampling_frequency_acc = np.round(len(time_acc)/(time_acc[-1]))
    
    #Filtering
        filter_acc =filter_data(acc,fs=sampling_frequency_acc,fc=2)
        filter_gyr = filter_data(gyr, fs=sampling_frequency_gyr, fc=4)
    
    #Cutting
        acc_cut, gyr_cut = cut_data(filter_acc,filter_gyr,samp_freq_acc)
    
    #PCA
        data_gyr = scale(gyr_cut)
        pca_gyr = decomposition.PCA(n_components=1)
        pca_gyr.fit(data_gyr)
        trans_gyr = pca_gyr.transform(data_gyr)
        trans_df_gyr = pd.DataFrame(trans_gyr)
        data_acc = scale(acc_cut)
        pca_acc = decomposition.PCA(n_components=1)
        pca_acc.fit(data_acc)
        trans_acc = pca_acc.transform(data_acc)
        trans_df_acc = pd.DataFrame(trans_acc)
    
    #Segmenting
        peaks, diffpeaks = seg_data_acc(trans_df_acc.iloc[1:,0].values.astype(float))
        cycle_list_acc = []
        j=0
        for i in peaks:
            cycle_list_acc.append(trans_df_acc.iloc[j:i+1,0].values.astype(float))
            j=i+1
        # print(cycle_list_acc)

    #Resampling
        df1 = pd.DataFrame(cycle_list_acc)
        df2 = pd.DataFrame()
        for i in range(df1.shape[1]):
            for j in range(df1.shape[0]):
                try :
                    df2 = signal.resample(df1.iloc[0:j,0:i],50)
                except :
                    pass
        print(df2.shape)
        plt.plot(df2)
        plt.show()