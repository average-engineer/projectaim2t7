#%% Importing Modules
import matplotlib.pyplot as plt
import numpy as np
from scipy.fft import fft, fftfreq, rfft, rfftfreq # For FFT
import pandas as pd
from sklearn.preprocessing import scale
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from mlxtend.plotting import plot_confusion_matrix
from sklearn.metrics import confusion_matrix
from keras.callbacks import EarlyStopping, ModelCheckpoint

# Importing function for data preprocessing
from dataPreprocess import dataPreprocess


#%% Neural Network

#%% Obtaining pre-processed data
balancedData,label = dataPreprocess()

f=balancedData.iloc[:,2:202]
g=balancedData.iloc[:,202]


X = f
y = g

# Number of Folds for cross-validation
numFolds = 10

#Initiating an empty array to store the accuracy and loss for each fold

acu = np.zeros(numFolds)
los = np.zeros(numFolds)
count = 0

kf = KFold(n_splits=numFolds, shuffle = True)
kf.get_n_splits(X)
# print(kf)
for train_index, test_index in kf.split(X):
#     print("TRAIN:", train_index, "TEST:", test_index)
#     print(train_index)
    X_train, X_test = X.iloc[train_index,:], X.iloc[test_index,:]
    y_train, y_test = y[train_index], y[test_index]
#     print(X_train, y_train)
#     print(X_train.shape)
#    print(y_test.value_counts())
#    print(y_train.value_counts())
    

    #y_pred = forest.predict(X_test[:])
    #print('Test Accuracy: %.3f' % accuracy_score(y_test, y_pred))

    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(200,)), 
        tf.keras.layers.Dense(136, activation ='relu'),
        #tf.keras.layers.Dense(93, activation ='relu'),
        # tf.keras.layers.Dense(5, activation ='relu'),
        tf.keras.layers.Dense(3, activation ='softmax')
        ])
        
    model.compile(optimizer = 'adam', loss = 'sparse_categorical_crossentropy', metrics = ['accuracy'])

    # train the neural network for 5 training epochs
    epochNum = 100 # Number of epochs
    
    #batchSize = 10
    
    # Without epoch early stopping
    # history = model.fit(X_train, y_train, epochs = epochNum, validation_data = (X_test, y_test), verbose = 1);
        
    # With epoch early stopping
    callback_a = ModelCheckpoint(filepath='my_best_mode.hdf5', monitor='val_loss', save_best_only=True, save_weights_only=True)
    callback_b = EarlyStopping(monitor='val_loss', mode='min', patience=10, verbose=1)
    history = model.fit(X_train, y_train, validation_data = (X_test, y_test), epochs = epochNum, callbacks=[callback_a, callback_b])
    
    loss, accuracy = model.evaluate(X_test, y_test, verbose=1);
    
    # Appending the loss and accuracy values in a numpy array
    acu[count] = accuracy
    los[count] = loss
    count = count+1

#print(los, '\n\n', acu)
# Printing mean losses and accuracies across all folds
print('Loss=', np.mean(los))
print('accuracy=', np.mean(acu))


# Plotting Accuracies and Losses
 
def plot_learningCurve(history, epochs):
    #Plot training & validation accuracy values
    fig = plt.figure(figsize = (15,10))
    epoch_range = range(1, epochs+1)
    plt.plot(epoch_range, history.history['accuracy'])
    plt.plot(epoch_range, history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc = 'upper left')
    plt.show()
    plt.grid()

    #Plot training & validation loss values
    fig = plt.figure(figsize = (15,10))
    plt.plot(epoch_range, history.history['loss'])
    plt.plot(epoch_range, history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Val'], loc = 'upper left')
    plt.show()
    plt.grid()
    
    
# plot_learningCurve(history, epochNum) # Plotting without epoch stopping

# Plotting with epoch stopping

# Accuracy
fig = plt.figure(figsize = (15,10))
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Training Accuracy', 'Validation Accuracy'], loc='lower right',prop = {"size": 20} )
plt.grid()
plt.show()



# Loss
fig = plt.figure(figsize = (15,10))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Training Loss', 'Validation Loss'], loc='upper right',prop = {"size": 20})
plt.grid()
plt.show()



#%% Checking Data Balance
print(y_test.value_counts())
print(y_train.value_counts())


#%% Computing Confusion Matrix
y_pred = model.predict(X_test)
classes_x = np.argmax(y_pred,axis=1)

mat = confusion_matrix(y_test, classes_x)
plot_confusion_matrix(conf_mat=mat, class_names=label.classes_, show_normed=False, figsize = (10,10))


#%% Performance Parameters from Confusion Matrix

# For each label(gait), there will be true positive, true negative, false positive, false negative

# Empty Lists for all positives and negatives
TP = np.empty((3))
TN = np.empty((3))
FP = np.empty((3))
FN = np.empty((3))
accuracy = np.empty((3))
precision = np.empty((3))
recall = np.empty((3))
f_score = np.empty((3))
for i in range(0,3):
    TP[i] = mat[i,i]
    TN[i] = mat.trace() - mat[i,i]
    FP[i] = mat[:,i].sum() - mat[i,i]
    FN[i] = mat[i,:].sum() - mat[i,i]
    accuracy[i] = (TP[i] + TN[i])/(TP[i] + TN[i] + FP[i] + FN[i])
    precision[i] = (TP[i])/(TP[i] + FP[i])
    recall[i] = (TP[i])/(TP[i] + FN[i])
    f_score[i] = 2*(precision[i])*(recall[i])/(precision[i] + recall[i])

f_scoreModel = f_score.mean()
