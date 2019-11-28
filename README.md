# Deep-Learning-for-Human-Presence-Detection
This repository contains codes for RNN and CNN-based models developed for human presence and movement detection using WiFi Control data. The classes of the classification problem are
1. Empty room
2. Stationary human present
3. Moving human present. 

'CNN_2DFFT_Alexnet.py' contains the code of AlexNet convolutional neural network (CNN) architecture (https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf) for the classification of human presence and movement detection through image classification. This code is wtitten in Python and Keras, which is a high-level API run on top of TensorFlow.

'RNN_LSTM_3_Classes_FFT_Keras.py' - Contains the code of solving the classification of human presence and movement using a recurrent neural network (RNN). This code is written in Python and Keras. We use two layers of long short-term memory (LSTM) cells to mitigate the vanishing and exploding gradient problem. The number of time steps in the RNN is 40.

'RNN_LSTM_3_Classes_FFT_Tensorflow.py' - Contains the code of solving the classification of human presence and movement using a RNN. This code is wtitten in Python and TensorFlow, without using Keras. We use one layer of long LSTM cells here.

'RNN_LSTM_3_Classes_FFT_12RNN_Train_Keras.py' - Contains the code of soving the classification of human presence and movement using a network of 12 parallel RNNs, instead of a single RNN. In this parallel RNN architecture, to obtain the final classification outcome in the Test set, the classification outputs of the 12 RNNs are combined using voting. For each RNN in this parallel model, we use two LSTM layers. This code is wtitten in Python and Keras.
