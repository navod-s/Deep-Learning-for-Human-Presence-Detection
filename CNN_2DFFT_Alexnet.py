# -*- coding: utf-8 -*-
"""
This code is written in Python and Keras.

This code implements the training of AlexNet CNN Architecture for our application of human presence and 
movement detection. The classes of the classification problem are
1. Empty room
2. Stationary human present
3. Moving human present. 
The AlexNet paper can be found in 
"https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf".


Navod Suraweera, Macquarie University
"""


from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.models import Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import seaborn as sns
sns.set()
from keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
test_datagen = ImageDataGenerator(rescale=1./255)

#Train and Validation set data generators
train_generator = train_datagen.flow_from_directory('./Train/', class_mode='categorical', subset='training')
valid_generator = train_datagen.flow_from_directory('./Train/', class_mode='categorical', subset='validation')

#Test set data generators
test_generator = test_datagen.flow_from_directory('./Test/', class_mode='categorical')


model = Sequential()

# Layer 1 - Convolution and max pooling
model.add(Convolution2D(filters=96, input_shape=(256,256,3), kernel_size=(11,11), strides=(4,4)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 2 - Convolution and max pooling
model.add(Convolution2D(256, 5, 5, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 3 - Convolution and max pooling
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(512, 3, 3, border_mode='same'))
model.add(Activation('relu'))

# Layer 4 - Convolution and max pooling
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(Activation('relu'))

# Layer 5 - Convolution and max pooling
model.add(ZeroPadding2D((1,1)))
model.add(Convolution2D(1024, 3, 3, border_mode='same'))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

# Layer 6 - Fully connected
model.add(Flatten())
model.add(Dense(3072, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 7- Fully connected
model.add(Dense(4096, init='glorot_normal'))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Layer 8 - Output layer
model.add(Dense(3, init='glorot_normal'))
model.add(Activation('softmax'))


optim = Adam(lr=0.0022, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(optimizer = optim, loss='categorical_crossentropy', metrics=['accuracy'])

#Saving the model parameters to be used for testing
filepath = "best_model.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
call_backs_list = [checkpoint]

max_epochs = 15
history = model.fit_generator(
    train_generator,
    steps_per_epoch = 100,
    epochs = max_epochs,
    validation_data = valid_generator,
    callbacks = call_backs_list,
    validation_steps = 50.
)


