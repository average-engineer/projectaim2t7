import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import owncloud
from datetime import datetime
import numpy as np
from scipy import signal
import re

def main():
    public_link = 'https://rwth-aachen.sciebo.de/s/ZpRI67lfqD27VlK?path=%2F'
    folder_password = 'CIE_2021'
    oc = owncloud.Client.from_public_link(public_link, folder_password=folder_password)
    # save_location = '/home/raj/Documents/modules/computational-intelligence-in-engineering/git/projectaim2t7/'
    # date_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    # fullsave_location = os.path.join (save_location, date_time)
    # os.makedirs(fullsave_location)
    #sub = np.arange(215)
    #sub = list(sub.astype(str))
    # print(sub)
    subject_id = ['215','216','217']
    lsc = []
    gai=[]
    time=[]
    xa = []
    ya = []
    za = []
    xg = []
    yg = []
    zg = []
    files = oc.list('/Smartphone3/', depth = 'infinity')
    for file in files:
        if not file.is_dir(): #and sub != 242 :
            if any(x in file.get_path() for x in subject_id) :
                filename = os.path.splitext(file.get_name())[0]
                ext = os.path.splitext(file.get_name())[1]
                if ext == '.csv':        
                    parent = os.path.basename(os.path.normpath(file.get_path()))
                    gait = parent.split("_")
                    #print(gait)
                    if parent != 'meta':
                        if not re.search(r'red', gait[1]):
                            content=oc.get_file_contents(file.get_path() + '/' + file.get_name())
                            csv = pd.read_csv(BytesIO(content))                           
                            lencsv = len(csv)-1
                            t = csv.iat[lencsv,0]
                            freq = (float(lencsv)+1.0)/t
                            lsc.append(freq)
                            for k in csv.iterrows():
                                dat = k[1]
                                if filename == 'Accelerometer' or filename == 'accelerometer': 
                                    xa.append(dat[1])
                                    ya.append(dat[2])
                                    za.append(dat[3])
                                else:
                                    xg.append(dat[1])
                                    yg.append(dat[2])
                                    zg.append(dat[3])

    dfac = pd.DataFrame({'X(m/s^2)': xa,'Y(m/s^2)': ya,'Z(m/s^2)': za})
    dfg = pd.DataFrame({'X(rad/s)': xg,'Y(rad/s)': yg,'Z(rad/s)': zg})

    print(dfac) 
    print(dfg) 
    #print('\nAverage frequency: ', lsc, 'Hz\n')
    

if __name__ == '__main__':
    main()