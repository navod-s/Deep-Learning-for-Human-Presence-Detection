"""
This code is written in Python and Keras.

This code implements the training of an RNN Architecture with 12 Parallel RNNs for our application 
of human presence and movement detection. The classes of the classification problem are
1. Empty room
2. Stationary human present
3. Moving human present. 
In this parallel RNN architecture, to obtain the final classification outcome in the Test set, the 
classification outputs of the 12 RNNs are combined using voting.


Navod Suraweera, Macquarie University
"""

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from keras.optimizers import adam
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split

import numpy as np
from datetime import datetime
import scipy.io as sio

startTime=datetime.now()
num_classes = 3
num_Sub=62
# Loading input data as a .mat file
mat = sio.loadmat('RNN_FFT_12ele_train_no_farad.mat')


#The number of time steps (T) in the RNN.
time_steps=int(mat['time_steps']);

#Training data
train_data_1 = mat['train_set_FFT_P1']
train_data_2 = mat['train_set_FFT_P2']
train_data_3 = mat['train_set_FFT_P3']
train_data_4 = mat['train_set_FFT_P4']
train_data_5 = mat['train_set_FFT_P5']
train_data=np.concatenate((train_data_1,train_data_2,train_data_3,train_data_4,train_data_5),axis=0)

#Length of each training data sample
packet_len=np.shape(train_data)[1]

# Training data labels 
out_train=train_data[:,packet_len-num_classes:packet_len]


num_train_batches=np.shape(train_data)[0]//time_steps

y_train=np.zeros((num_train_batches,num_classes))

X=np.array(range(12))

for x in X:
    RNN_ind=x+1
    X_train=train_data[:,(RNN_ind-1)*num_Sub:(RNN_ind)*num_Sub]      
    train_batches=np.zeros((num_train_batches,time_steps,num_Sub))   
    
    for i in range (num_train_batches):
        train_batches[i,:,:]=X_train[i*time_steps:(i+1)*time_steps,:]
        y_train[i,:]=out_train[time_steps*i,:]
     
    Train_X, Val_X, Train_Y, Val_Y = train_test_split(train_batches, y_train, test_size = 0.25)   
    
    # Number of hidden node in each LSTM cell. For each RNN in this parallel model, we use two LSTM layers.
    num_hidden = 128

    model = Sequential()
    model.add(LSTM(num_hidden, activation='relu', return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(num_hidden, activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation='softmax'))    
    opt = adam(lr=0.001, decay=1e-6)
    
    model.compile(
        loss='categorical_crossentropy',
        optimizer=opt,
        metrics=['accuracy'],
    )
    
    filepath="./RNN_12_Faraday_Val/RNN_"+str(RNN_ind)+"_weights-improvement-{epoch:02d}.hdf5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]
    
    model.fit(Train_X,Train_Y,epochs=10, validation_data=(Val_X, Val_Y),callbacks=callbacks_list, verbose=1)
    
print(datetime.now() - startTime) 
