
# Importing Modules
import matplotlib.pyplot as plt
import numpy as np

## Obtaining all pertinent data for the mentioned subject numbers
from csvread import csvread # importing the function for reading all the csv files
accX,accY,accZ,gyrX,gyrY,gyrZ,timeVector,sampFreq = csvread()

