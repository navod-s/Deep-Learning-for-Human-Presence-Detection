"""
This code is wtitten in Python and TensorFlow without using Keras.

This code implements the training of an RNN Architecture for our application 
of human presence and movement detection. We use a single-LSTM layer in this network.

The classes of the classification problem are
1. Empty room
2. Stationary human present
3. Moving human present. 

Navod Suraweera, Macquarie University
"""

import tensorflow as tf
import numpy as np
from datetime import datetime
import scipy.io as sio
#
tf.reset_default_graph()
startTime=datetime.now()

#The number of classes in the classifier
num_classes = 3

#Data input as a .mat file
mat_train = sio.loadmat('RNN_FFT_12ele_train.mat')
mat_test = sio.loadmat('RNN_FFT_12ele_test.mat')
"""
The length of the sequence input into the RNN in-terms of the number of time steps (T)
For each T value we generate test and train data batches, which are contained 
in the input data mat file.
"""
#The number of time steps (T) in the RNN.
time_steps=int(mat_train['time_steps']);

train_data_1 = mat_train['train_set_FFT_P1']
train_data_2 = mat_train['train_set_FFT_P2']
train_data_3 = mat_train['train_set_FFT_P3']
train_data_4 = mat_train['train_set_FFT_P4']
train_data_5 = mat_train['train_set_FFT_P5']
train_data=np.concatenate((train_data_1,train_data_2,train_data_3,train_data_4,train_data_5),axis=0)

test_data_1 = mat_test['test_set_FFT_1']


packet_len=np.shape(train_data)[1]

#Number of input data batches used for training and testing 
num_train_batches=np.shape(train_data)[0]//time_steps
num_test_batches_1=np.shape(test_data_1)[0]//time_steps

"""
Generate train and test data batches
"""
X_train=train_data[:,0:packet_len-num_classes]
out_train=train_data[:,packet_len-num_classes:packet_len]

X_test_1=test_data_1[:,0:packet_len-num_classes]
out_test_1=test_data_1[:,packet_len-num_classes:packet_len]



train_batches=np.zeros((num_train_batches,time_steps,packet_len-num_classes))
test_batches_1=np.zeros((num_test_batches_1,time_steps,packet_len-num_classes))

y_train=np.zeros((num_train_batches,num_classes))
y_test_1=np.zeros((num_test_batches_1,num_classes))

for i in range (num_train_batches):
    train_batches[i,:,:]=X_train[i*time_steps:(i+1)*time_steps,:]
    y_train[i,:]=out_train[time_steps*i,:]
    
for i in range (num_test_batches_1):
    test_batches_1[i,:,:]=X_test_1[i*time_steps:(i+1)*time_steps,:]
    y_test_1[i,:]=out_test_1[time_steps*i,:]
    

learning_rate = 0.001
training_steps =120
display_step = 50
num_input = packet_len-num_classes 
lambda_loss = 0.0001
# Number of hidden node in each LSTM cell
num_hidden = 200
num_layers = 3;

    
X = tf.placeholder("float", [None, time_steps, num_input])
Y = tf.placeholder("float", [None, num_classes]) 

#Initialization of weights ans biases
weights = {
    'out': tf.Variable(tf.random_normal([num_hidden, num_classes]))
}
biases = {
    'out': tf.Variable(tf.random_normal([num_classes]))
}

"""
Definition LSTM RNN graph using Tensorflow
"""

    
def RNN(x, weights, biases):
    x = tf.unstack(x, time_steps, 1)
    lstm_cell =tf.nn.rnn_cell.BasicLSTMCell(num_hidden, state_is_tuple=True)
    outputs, states = tf.nn.static_rnn(lstm_cell, x, dtype=tf.float32)

    return tf.matmul(outputs[-1], weights['out']) + biases['out']

logits = RNN(X, weights, biases)
prediction = tf.nn.softmax(logits)

prediction_int=tf.argmax(prediction, 1)
true_int=tf.argmax(Y, 1)

# Definition of the loss function and the optimizer used for training the RNN 
l2 = lambda_loss * sum(tf.nn.l2_loss(tf_var) for tf_var in tf.trainable_variables())

loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
    logits=logits, labels=Y))+l2
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss_op)
#train_op = optimizer.minimize(loss_op)

correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()


"""
This is where the RNN is actually built, trained and tested (inside a Tensorflow session),
based on the above definitions of the graph
"""
with tf.Session() as sess:

    sess.run(init)

    for step in range(1, training_steps+1):

        batch_x = train_batches
        batch_y= y_train

        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:

            loss, acc = sess.run([loss_op, accuracy], feed_dict={X: batch_x,
                                                                 Y: batch_y})
            print("Step " + str(step) + ", Minibatch Loss= " + \
                  "{:.4f}".format(loss) + ", Training Accuracy= " + \
                  "{:.3f}".format(acc))

    print("Training Finished....")

    batch_x_test_1 = test_batches_1
    batch_y_test_1= y_test_1
    int_pred=sess.run(prediction_int, feed_dict={X: batch_x_test_1, Y: batch_y_test_1})  
    int_true=sess.run(true_int, feed_dict={X: batch_x_test_1, Y: batch_y_test_1})  
    
    from sklearn.metrics import confusion_matrix
    cm_3_Cl_SC_Norm = confusion_matrix(int_true, int_pred)
    
 #Testing using the trained RNN model   
    print("Testing Accuracy for Faraday:", \
        sess.run(accuracy, feed_dict={X: batch_x_test_1, Y: batch_y_test_1}))   
    


print(datetime.now() - startTime) 
