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
paths =glob.glob(r'./Data/Smartphone3/*')

for path in paths:
    name = re.split('/',path)
    experiment_name = name[-1]
    splitexp = re.split('_',experiment_name)
    subj=splitexp[0]
    gait=splitexp[1]
    if not re.search(r'red', gait) and re.search(r'215',subj) or re.search(r'216',subj) or re.search(r'217',subj):
        # re.search(r'215',subj) or re.search(r'217',subj)
        gyr_file = glob.glob(path+'/Gyroscope.csv')
        acc_file = glob.glob(path+'/Accelerometer.csv')
        data_gyr = pd.read_csv(gyr_file[0], sep=",")
        gyr = data_gyr.iloc[2:,1:4].values.astype(float)
        data_acc = np.loadtxt(acc_file[0], delimiter=',', skiprows=1)
        acc = data_acc[:,1:4]
        time = data_acc[:,0]
        samp_freq = np.round(len(time)/(time[-1]))
        sampling_frequency = np.round(len(time)/(time[-1]))
        filter_acc =filter_data(acc,fs=sampling_frequency,fc=4)
        filter_gyr = filter_data(gyr, fs=sampling_frequency, fc=4)
        data_gyr = scale(filter_gyr)
        pca_gyr = decomposition.PCA(n_components=1)
        pca_gyr.fit(data_gyr)
        trans_gyr = pca_gyr.transform(data_gyr)
        trans_df_gyr = pd.DataFrame(trans_gyr)
        data_acc = scale(filter_acc)
        pca_acc = decomposition.PCA(n_components=1)
        pca_acc.fit(data_acc)
        trans_acc = pca_acc.transform(data_acc)
        trans_df_acc = pd.DataFrame(trans_acc)
        plt.plot(trans_df_acc)
        plt.plot(trans_df_gyr)
        plt.show()