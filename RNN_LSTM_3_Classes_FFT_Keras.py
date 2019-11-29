"""
This code is written in Python and Keras.

This code implements the training of an RNN Architecture for our application 
of human presence and movement detection. In the RNN, we use two LSTM layers.

The classes of the classification problem are
1. Empty room
2. Stationary human present
3. Moving human present. 

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
#
#tf.reset_default_graph()

startTime=datetime.now()

#The number of classes in the classifier
num_classes = 3



mat = sio.loadmat('RNN_FFT_12ele_train.mat')

"""
The length of the sequence input into the RNN in-terms of the number of time steps (T)
For each T value we generate test and train data batches, which are contained 
in the input data mat file.
"""
time_steps=int(mat['time_steps']);
train_data_1 = mat['train_set_FFT_P1']
train_data_2 = mat['train_set_FFT_P2']
train_data_3 = mat['train_set_FFT_P3']
train_data_4 = mat['train_set_FFT_P4']
train_data_5 = mat['train_set_FFT_P5']
train_data=np.concatenate((train_data_1,train_data_2,train_data_3,train_data_4,train_data_5),axis=0)



packet_len=np.shape(train_data)[1]

#Number of input data batches used for training and testing 
num_train_batches=np.shape(train_data)[0]//time_steps

"""
Generate train and test data batches
"""
X_train=train_data[:,0:packet_len-num_classes]
out_train=train_data[:,packet_len-num_classes:packet_len]

train_batches=np.zeros((num_train_batches,time_steps,packet_len-num_classes))

y_train=np.zeros((num_train_batches,num_classes))

for i in range (num_train_batches):
    train_batches[i,:,:]=X_train[i*time_steps:(i+1)*time_steps,:]
    y_train[i,:]=out_train[time_steps*i,:]
        
Train_X, Val_X, Train_Y, Val_Y = train_test_split(train_batches, y_train, test_size = 0.4, random_state = 0)
    

learning_rate = 0.001
display_step = 50
num_input = packet_len-num_classes 
lambda_loss = 0.0001
# Number of hidden node in the LSTM cell
num_hidden = 128
num_layers = 3;

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

filepath="./RNN_1NW/val 0.4/weights-improvement-{epoch:02d}.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
callbacks_list = [checkpoint]

model.fit(Train_X,Train_Y,epochs=15, validation_data=(Val_X, Val_Y),callbacks=callbacks_list, verbose=1)


    
