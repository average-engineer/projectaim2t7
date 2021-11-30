import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
import glob
import re
import pandas as pd
from scipy.signal.ltisys import TransferFunctionDiscrete
from sklearn.preprocessing import scale
from sklearn import decomposition


def to_str(var):
    return str(list(np.reshape(np.asarray(var), (1, np.size(var)))[0]))[1:-1]

def filter_data(sensordata, fs=200, fc=5):
    sensordata_filt = np.zeros(sensordata.shape)
    
    w = fc / (fs / 2)
    b, a = signal.butter(fc,w, 'low')
    
    sensordata_filt[:,0] = signal.filtfilt(b,a, sensordata[:,0])
    sensordata_filt[:,1] = signal.filtfilt(b,a, sensordata[:,1])
    sensordata_filt[:,2] = signal.filtfilt(b,a, sensordata[:,2])
    
    return sensordata_filt
    
def cut_data(acc_filt, gyr_filt):
    acc_abs = np.linalg.norm(acc_filt,axis=1)
    

    peaks, _ = signal.find_peaks(acc_abs,height=11)
    diff_peaks =np.diff(peaks)
    gap1  = np.argmax(diff_peaks[:10])
    gap2  = np.argmax(diff_peaks[-10:])
    gap2  = int(gap2 + np.shape(diff_peaks)-10)

    
    acc_cut = acc_filt[peaks[gap1+1]:peaks[gap2],:]
    gyr_cut = gyr_filt[peaks[gap1+1]:peaks[gap2],:]
    
    return acc_cut, gyr_cut

def seg_data_acc(trans_df_acc):
    peaks, _ = signal.find_peaks(trans_df_acc)
    diff_peaks =np.diff(peaks)

    return peaks,diff_peaks


column_names = ['trial', 'sensor', 'gait', 'samples']
training_data_acc = pd.DataFrame(columns=column_names)
training_data_gyr = pd.DataFrame(columns=column_names)
sample_list_acc=[]
sample_list_gyr=[]
df = pd.DataFrame()
gait_list = []
file_list =[]
paths =glob.glob(r'./Data/Smartphone3/*')

for path in paths:
    name = re.split('/',path)
    experiment_name = name[-1]
    splitexp = re.split('_',experiment_name)
    subj=splitexp[0]
    gait=splitexp[1]
    if not re.search(r'red', gait):
        # re.search(r'215',subj) or re.search(r'217',subj)
        try:
            file_list.append(experiment_name)
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
            #  print(experiment_name)
            if sampling_frequency_acc > 10:        
        # Filtering
                filter_acc =filter_data(acc,fs=sampling_frequency_acc,fc=2)
                filter_gyr = filter_data(gyr, fs=sampling_frequency_gyr, fc=1)
            # Cutting
                acc_cut, gyr_cut = cut_data(filter_acc,filter_gyr)
                
            # PCA
                data_gyr = scale(gyr_cut)
                pca_gyr = decomposition.PCA(n_components=1)
                pca_gyr.fit(data_gyr)
                trans_gyr = pca_gyr.transform(data_gyr)
                sample_list_gyr.append(trans_gyr)
                data_acc = scale(acc_cut)
                pca_acc = decomposition.PCA(n_components=1)
                pca_acc.fit(data_acc)
                trans_acc = pca_acc.transform(data_acc)
                sample_list_acc.append(trans_acc)
                ga = re.split('0',gait)
                gait_list.append(ga)

        except:
            pass
# for i in range(len(sample_list_acc)):
#     print(len(sample_list_acc[i]))
try:
        
    for k in range(len(file_list)):
        training_data_acc['trial'] = training_data_acc['trial'].astype('object')
        training_data_acc.at[k,'trial'] = sample_list_acc[k].transpose()
        training_data_acc.at[k,'sensor'] = 'Accelerometer'
        training_data_acc.at[k, 'gait'] = gait_list[k][0]
        training_data_gyr['trial'] = training_data_gyr['trial'].astype('object')
        training_data_gyr.at[k,'trial'] = sample_list_gyr[k].transpose()
        training_data_gyr.at[k,'sensor'] = 'Gyroscope'
        training_data_gyr.at[k, 'gait'] = gait_list[k][0]
except:
    pass
trilist1=[]
for i in range(len(training_data_acc)):
    trilist1.append((training_data_acc.at[i,'trial'][0]).tolist())
    training_data_acc.loc[i,'samples'] = trilist1[i]
training_data_acc.drop(training_data_acc.columns[0], axis=1, inplace=True)
trilist2=[]
for i in range(len(training_data_gyr)):
    trilist2.append((training_data_gyr.at[i,'trial'][0]).tolist())
    training_data_gyr.loc[i,'samples'] = trilist2[i]
training_data_gyr.drop(training_data_gyr.columns[0], axis=1, inplace=True)
frames = [training_data_acc,training_data_gyr]
result = pd.concat(frames)
# print(result)
result.to_csv('training_data.csv',index=False, encoding="utf-8",escapechar='\\', doublequote=False)

        # Segmenting
            # peaks_acc, diffpeaks_acc = seg_data_acc(trans_df_acc.iloc[1:,0].values.astype(float))
        # cycle_list_acc = []
        # cycle_list_gyr = []
        # j=0
        # for i in peaks_acc:
            # cycle_list_acc.append(trans_df_acc.iloc[j:i+1,0].values.astype(float))
            # j=i+1
# 
        # peaks_gyr, diffpeaks_gyr = seg_data_acc(trans_df_gyr.iloc[1:,0].values.astype(float))
        # m=0
        # for n in peaks_gyr:
            # cycle_list_gyr.append(trans_df_gyr.iloc[m:n+1,0].values.astype(float))
            # m=n+1
        # print(cycle_list_acc)
        # print(cycle_list_gyr)
# 
    # Resampling
        # df1_acc = pd.DataFrame(cycle_list_acc)
        # df2_acc = pd.DataFrame()
        # for i in range(df1_acc.shape[1]):
            # for j in range(df1_acc.shape[0]):
                # try :
                    # df2_acc = signal.resample(df1_acc.iloc[0:j,0:i],1000)
                # except :
                    # pass
        # print(df2_acc.shape)
        # plt.plot(df2_acc)
        # plt.show()
        # df1_gyr = pd.DataFrame(cycle_list_gyr)
        # df2_gyr = pd.DataFrame()
        # for i in range(df1_gyr.shape[1]):
            # for j in range(df1_gyr.shape[0]):
                # try :
                    # df2_gyr = signal.resample(df1_gyr.iloc[0:j,0:i],100)
                # except :
                    # pass
        # print(df2_gyr.shape)
        # plt.plot(df2_gyr)
        # plt.show()