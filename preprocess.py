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
    peaks, _ = signal.find_peaks(trans_df_acc) # Returns indices of peaks 
    diff_peaks =np.diff(peaks)

    return peaks,diff_peaks




paths =glob.glob(r'./Data/Smartphone3/*')

for path in paths:
    name = re.split('/',path)
    experiment_name = name[-1]
    splitexp = re.split('_',experiment_name)
    subj=splitexp[0]
    gait=splitexp[1]
    if not re.search(r'red', gait) and re.search(r'216',subj):
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
        # print('Duration of Acc_exp:', time_acc[-1], 'Duration of Gyr_exp:', time_gyr[-1])
        filter_acc =filter_data(acc,fs=sampling_frequency_acc,fc=2)
        filter_gyr = filter_data(gyr, fs=sampling_frequency_gyr, fc=4)
        acc_cut, gyr_cut = cut_data(filter_acc,filter_gyr,samp_freq_acc)
        # print(gyr_cut)
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
        peaks, diffpeaks = seg_data_acc(trans_df_acc.iloc[1:,0].values.astype(float))
        cycle_list_acc = []
        j=0
        for i in peaks:
            cycle_list_acc.append(trans_df_acc.iloc[j:i+1,0].values.astype(float))
            j=i+1
        data_cycle_1 = pd.DataFrame()
        data_cycle_2 = pd.DataFrame()
        for i in range(len(cycle_list_acc)):
            data_cycle_1[i] = cycle_list_acc
            for j in data_cycle_1:
                data_cycle_2 = data_cycle_1[j]
print(data_cycle_2.iloc[0,:])
        # plt.plot(data_cycle_2.iloc[0])
        # plt.show()
        # data_cycle_2.iloc[:].plot()
        #     plt.plot(data_cycle_2)
        #     plt.show()
        # res_cyc = signal.resample(df_cyc[0],1750)
        # plt.plot(res_cyc)
        # # plt.plot(res_acc)
        # # plt.plot(res_gyr)
        # #  res_acc = signal.resample(trans_df_acc[0],1750)
        # # res_gyr = signal.resample(trans_df_gyr[0],int(time_gyr[-1]))
        # plt.show()