import matplotlib.pyplot as plt
import pandas as pd
import os
from io import BytesIO
import owncloud
from datetime import datetime
import numpy as np
from scipy import signal
import re


## ******************* Desired Output of the Function *************************************
    # -> Sampling frequencies of all csv files being read
    # -> Time vectors of all csv files being read
    # -> X,Y,Z readings of acclerometer sensors for all subject files being read
    # -> X,Y,Z readings of gyroscope sensors for all subject files being read
    # -> All data is returned as lists





#***************** Structure of the Remote storage of the dataset:**************************
    # Home Folder -> Smartphone1
    #             -> Smartphone2
    #             -> Smartphone3 
    #                           -> subject<Subject#>_<GaitType><Trial#>
    #                                                                 -> Accelerometer.csv
    #                                                                 -> Gyroscope.csv
    #             -> Smartphone4
#********************************************************************************************


def csvread(): # All the data parsing happens in a main function
    # Obtaining the datasets from the remote storage (Sceibo)
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
    subject_id = ['215','216','217'] # list containing the subject numbers as strings
    # subject_id is used to detect the subject #s in the subfolders
    
    # Initialising an empty directory for obtaining the sampling frequnecy and timevector
    # of each csv file
    # In this directory, the sampling frequency of the file as the dictionary key and the 
    # corresponding timevector of the csv file as the dictionary value
    freqdic = {} # Empty Dictionary
    
    lsc = [] # Empty List
    xa = [] # Empty list for storing all X acceleration values
    ya = [] # Empty list for storing all Y acceleration values
    za = [] # Empty list for storing all Z acceleration values
    xg = [] # Empty list for storing all X gyroscope values
    yg = [] # Empty list for storing all X gyroscope values
    zg = [] # Empty list for storing all X gyroscope values
    
    
    files = oc.list('/Smartphone3/', depth = 'infinity') # files is a list 
    # Each element of the list files is basically an object which contains info
    # on the files contained in the remote directory we are in
    # At this point in the code, we are inside the Smartphone 3 folder 
    # Inside the folder now we need to acccess each of the subjects files 
    for file in files: # Iterating through all the subject files in the Smartphone 3 folder
        if not file.is_dir(): #and sub != 242 :
            if any(x in file.get_path() for x in subject_id) : # the subject IDs (#s) are matched and we enter into the respective
                                                               # folders which return a match with x (which is iterating through
                                                               # the subject IDs)
                ## At this point, we have enetered the folder whose subject # matches with x (subject ID)
                # Splitting the filenames into names and extensions
                filename = os.path.splitext(file.get_name())[0] # Names are extracted from the first element (Accelerometer/Gyroscope)
                ext = os.path.splitext(file.get_name())[1] # File extensions are extracted from the second element
                # Ensuring we read only csv files
                # print(filename,ext)
                if ext == '.csv':        
                    parent = os.path.basename(os.path.normpath(file.get_path())) # Storing the name of the parent folders
                    # Parent Folder Names-> subject<Subject#>_<GaitType><Trial#>
                    # parent variable references to a string
                    # The string is split into the subject numbers and the gait type  trial number
                    # subject<Subject#>_<GaitType><Trial#> -> ['subject<Subject#>','<GaitType><Trial#>']
                    gait = parent.split("_") # gait is a list with strings as elements
                    # print(gait)
                    if parent != 'meta': # Ensuring no folder with the name 'meta' are read
                        # Our aim is to just differentiate between 3 gait types: Normal/Even,Upstairs,Downstairs
                        # All folders having information on Impaired walking are omitted
                        if not re.search(r'red', gait[1]): # All folders having the words 'red' in their names
                            content=oc.get_file_contents(file.get_path() + '/' + file.get_name()) 
                            csv = pd.read_csv(BytesIO(content)) # csv is a dataframe defined under the Pandas module
                            
                            # csv basically parses one entire csv file into itself
                            
                            # Structure of csv dataframe
                            # <RowIndex#(0)>  <csv 1st column header>  <csv 2nd column header>  <csv 3rd column header>
                            # <RowIndex#(1)>  <csv [1,1]>              <csv [1,2]>              <csv [1,3]>
                            # .............................
                            # .............................
                            # .............................
                            # .............................
                            # <RowIndex#(i)>  <csv [i,1]>              <csv [i,2]>              <csv [i,3]>
                            
                            lencsv = len(csv)-1 # Number of rows in the csv file
                            t = csv.iat[lencsv,0] # Last Time instant in a csv file
                            samp_freq = (float(lencsv)+1.0)/t # Sampling Frequency of the excel file being read
                            #lsc.append(freq)
                            
                            # Initiating numpy arrays for storing acceleration and gyroscope data for each
                            # csv file
                            # Each empty array will have number of rows equal to the number of rows in the csv file
                            # and only one column since we are storing only one dimension data at a time
                            accX = np.empty((len(csv),1)) 
                            accY = np.empty((len(csv),1))
                            accZ = np.empty((len(csv),1))
                            gyrX = np.empty((len(csv),1)) 
                            gyrY = np.empty((len(csv),1))
                            gyrZ = np.empty((len(csv),1))
                            
                            # Initialising numpy array for the time vector
                            time = np.empty((len(csv),1)) 
                            
                            
                            # Crude Fix: Check the filename first and then iterate through the csv ross:
                                # we will have 2 for loops in this case instead of one
                                
                            # Desired Structure of the lists
                            # xa = [{AccX of csv file 1},{AccX of csv file 2},.....,{AccX of csv file n}], n: # csv files
                            
                            # checking filename first
                            if filename == 'Accelerometer' or filename == 'accelerometer':
                                itera = 0 # iterating interger variable for acceleration values
                                
                                
                                # Iterating through the csv file rows
                                for k in csv.iterrows(): # iterating through the rows of the csv dataframe, k is a tuple (not an integer)
                                    # Structure of k: (<RowIndex#>, Info in all corresponding columns with headers)
                                    dat = k[1] # accessing only the column info (not the row index number)
                                    # if filename == 'Accelerometer' or filename == 'accelerometer': 
                                    #xa.append(dat[1]) # Adding all Acceleration X data to the empty xa array
                                    #ya.append(dat[2]) # Adding all Acceleration Y data to the empty ya array
                                    #za.append(dat[3]) # Adding all Acceleration Z data to the empty za array
                                    
                                    time[itera] = dat[0]
                                    accX[itera] = dat[1] # Filling up the acceleration X array
                                    accY[itera] = dat[2] # Filling up the acceleration Y array
                                    accZ[itera] = dat[3] # Filling up the acceleration Z array
                                    itera = itera + 1
                                    
                                    
                                # Appending the Acceleration arrays to the acceleration list
                                xa.append(accX) # Appending the accleeration X array to the list
                                ya.append(accY) # Appending the accleeration Y array to the list
                                za.append(accZ) # Appending the accleeration Z array to the list
                                
                                # Populating the dictionary with sampling frequency as the key and timevector as the value
                                freqdic[samp_freq] = time
                                    
                                    
                            else: # Gyroscope files
                                iterg = 0 # iterating interger variable for gyroscope values
                                
                                # Iterating through the csv file rows
                                for k in csv.iterrows():
                                    dat =  k[1]
                                    #xg.append(dat[1]) # Adding all Gyroscope X data to the empty xg array
                                    #yg.append(dat[2]) # Adding all Gyroscope Y data to the empty yg array
                                    #zg.append(dat[3]) # Adding all Gyroscope Z data to the empty zg array
                                    
                                    time[iterg] = dat[0]
                                    gyrX[iterg] = dat[1] # Filling up the gyroscope X array
                                    gyrY[iterg] = dat[2] # Filling up the gyrocsope Y array
                                    gyrZ[iterg] = dat[3] # Filling up the gyroscope Z array
                                    iterg = iterg + 1
                                    
                                # Doing the same fr=or the gyroscope data
                                xg.append(gyrX)
                                yg.append(gyrY)
                                zg.append(gyrZ)
                                    
                                # Populating the dictionary with sampling frequency as the key and timevector as the value
                                freqdic[samp_freq] = time                               
                                    
                                    
                            # Once all the arrays are filled, we append these arrays (numpy objects) to the empty
                            # lists defined earlier
                            
                            
                            # xa.append(accX) # Appending the accleeration X array to the list
                            # ya.append(accY) # Appending the accleeration Y array to the list
                            # za.append(accZ) # Appending the accleeration Z array to the list
                            
                            # Doing the same fr=or the gyroscope data
                            # xg.append(gyrX)
                            # yg.append(gyrY)
                            # zg.append(gyrZ)
                            
    
    
    # Extracting the values of the dictionaries (time vectors of all csv files) and sotring them in 
    # a seperate time list
    timeVecs = list(freqdic.values())
    
    # Extracting the keys of the dictionaries (sample frequencies of all csv files) and sotring them in 
    # a seperate sampling frequencies list
    sampFreqVec = list(freqdic.keys())
    
    
    
    # print(sampFreqVec) 
    # print('\n##############################\n')
    # print(gyrX)
    # print('\nAverage frequency: ', avg_freq, 'Hz\n')
    # df = pd.DataFrame(dic)
    # print(df) 
    # print(type(k))
    # print(dat)
    print('****END*****')
    
    # Returning all data
    return xa,ya,za,xg,yg,zg,timeVecs,sampFreqVec


# if __name__ == '__main__':
#    main()