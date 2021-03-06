import glob
import re
import numpy as np
import pandas as pd

#%% Global Variables
global coFreqAcc # Cut-off frequency for acceleration data
global coFreqGyr # Cut-off frequency for gyroscope data
coFreqAcc = 5 # Hz
coFreqGyr = 1 # Hz
#%% ******************* Desired Output of the Function *************************************
    # -> Sampling frequencies of all csv files being read
    # -> Time vectors of all csv files being read
    # -> X,Y,Z readings of acclerometer sensors for all subject files being read
    # -> X,Y,Z readings of gyroscope sensors for all subject files being read
    # -> All data is returned as lists




#%% ***************** Structure of the dataset:**************************
    # Home Folder -> Smartphone1
    #             -> Smartphone2
    #             -> Smartphone3 
    #                           -> subject<Subject#>_<GaitType><Trial#>
    #                                                                 -> Accelerometer.csv
    #                                                                 -> Gyroscope.csv
    #             -> Smartphone4
#********************************************************************************************



#%% Starting Data Parsing

def dataParser():
    
    print('Starting Data Parsing')
    
    # Part record wise sampling and Part subject wise
    # The subject numbers vary from 150 to 274
    # The first 101 subjects are taken for training and testing the neural network
    # The last 25 subject datasets are not fed into the neural network at all
    # and can be treated as a dataset which is completely unknown to the neural network
    # Once the model has been trained and tested, we can further validate our model
    # by just passing the 25 subject data as a new test data into the neural network.
    # In this way, we do both record wise and subject wise sampling.

    
    # Required Subject IDs -> Which Subject Data are taken
    startSub = 150
    endSub = 160
    numSub = endSub - startSub + 1
    subjID = list(np.linspace(startSub,endSub,numSub))
    # subjID = ['155','224','257']
    
    # Accesing all folders inside the Smartphone 3 folder
    folders = glob.glob(r'./ProjectAData/Data/Smartphone3/*')  # folders is a list containing all folder names as strings
    
    
    # Initialising empty dictionary for storing all acceleration and gyroscope data
    
    # The keys in the dictionary will be the sampling frequencies and experiment names corresponding to each csv file
    # The values in the dictionary will be the 2D arrays containing the entire csv data
    
    AccData = {} # Dictionary for Acceleration data
    GyrData = {} # Dictionary for gyroscope data
    
    # Empty List for storing experiment names
    exp = []
    
    # Empty List for storing subject names
    subjNum = []
    
    
    for folder in folders:
        
        # Ensuring the impaired folders are not accessed since they don't come under our scope
        # Any folder name having the series of characters 'red' in it will be skipped
        if re.search(r'red', folder) or re.search(r'cap', folder):
            continue # The next iteration is carried out
        # print(folder)
        
        # How each folder name (folder) look like: some examples
        # ./ProjectAData/Data/Smartphone3\subject196_normal02
        # ./ProjectAData/Data/Smartphone3\subject197_downstairs01
        # ./ProjectAData/Data/Smartphone3\subject197_downstairs02
        
        # We have to ensure that we don't access any impaired folders for any of the subjects
        
        # Splitting of folder names:
            # -> Experiment Type (Upstaris, downstairs)
            # -> Subject Number
        
        Split1 = re.split('/',folder) # Splitting each folder name
        Split1 = Split1[-1]
        Split2 = re.split('_',Split1)
        experiment = Split2[-1]
        experiment = re.split('0',experiment)
        experiment = experiment[0]
        
        # Standardising the experiment names
        if re.search(r'own',experiment):
            experiment = 'downstairs'
          
        if re.search(r'up',experiment):
            experiment = 'upstairs'  
            
        if re.search(r'or',experiment):
            experiment = 'normal' 
        
        
        
        Split2 = Split2[0]
        Split3 = re.split('t',Split2)
        subjectNum = Split3[-1]
        
        
        # Accesing the accelerometer and gyroscope files for the matching subject ids
        if float(subjectNum) in subjID and float(subjectNum) != 218 and float(subjectNum) != 222 and float(subjectNum) != 270: 
            # Subject 218 folder files are not in the right format
            # Subject 222 upstairs trial 01 folder doesn't have accelerometer file so the entire subject skipped
            # Subject 270 downstairs trial 02 folder doesn't have gyroscope file so the entire subject skipped
            gyr_file = glob.glob(folder+'/Gyroscope.csv')
            acc_file = glob.glob(folder+'/Accelerometer.csv')
            
            # gyr_file and acc_file are lists of size 1 with the onky element being the path name to the corresponding
            # gyroscope or acclerometer files
            # Example:
               # ['./ProjectAData/Data/Smartphone3\\subject216_normal02/Accelerometer.csv']
               # the above is one such acc_file when the subject ID of 216 matches
            
            # print(acc_file)
            # print(gyr_file)
            # print(subjectNum)
            
            # Extracting the accelerometer and gyroscope data
            data_gyr = pd.read_csv(gyr_file[0],sep = ",", header = None) # data_gyr is a dataframe
            # From the above dataframe, we are going to extract the X,Y,Z time series and assemble it into a 2D numpy array
            # The first row of the dataframe corresponds to the headers so that has to be skipped too
            # Remember -> Indexing starts from 0
            gyr = data_gyr.iloc[1:,0:4].values.astype(float) # gyr is a 2D numpy array containing the entire csv file
            # print(data_gyr)
            # print(gyr)
            
            data_acc = pd.read_csv(acc_file[0],sep = ",", header = None)
            acc = data_acc.iloc[1:,0:4].values.astype(float) # Columns 0,1,2,3 are read (Index 4 is left out, similar to range)
            # print(data_acc)
            # print(acc)
            
            # Computing Sampling Frequency:
                # Sampling Frequency = (# Datapoints in the csv)/(Experiment Time)
                # Experiment Time = Last element of the type vector
                # # Datapoints in the csv = number of rows in the 2D array
                
            sampFreqAcc = np.shape(acc)[0]/acc[-1,0]
            sampFreqGyr = np.shape(gyr)[0]/gyr[-1,0]
            
            # Due to some errors in data measurement, some sampling frequencies have blatantly wrong values (for eg. 0.3 Hz)
            # Since for using digital filters, we require the digital filter critical frequency to be between 0 and 1
            if sampFreqAcc < 2*coFreqAcc or sampFreqGyr < 2*coFreqGyr:
                continue # The entire subject is skipped
            
            
            # Dictionary keys have to be immutable only (like tuples)
            keyAcc = sampFreqAcc
            keyGyr = sampFreqGyr
            
            # Adding elements to the acceleration and gyroscope dictionaries
            AccData[keyAcc] = acc # The accelerometer file sampling frequency is the key and the csv data is the value
            GyrData[keyGyr] = gyr # The gyroscope file sampling frequency is the key and the csv data is the value
            
            # Appending experiment names to the experiment list
            exp.append(experiment)
            
            # Appending subject names to the subject list
            subjNum.append(subjectNum)
            
            # Printing the subject numbers and corresponding experiments whose files are accessed for reference
            print(subjectNum,experiment)
            
    # print(AccData)
        
    print('Ending Data Parsing')
    return AccData, GyrData, exp, subjNum