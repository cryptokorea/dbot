import tensorflow as tf
import random
import numpy as np

def MinMaxScaler(data):
    ''' Min Max Normalization

    Parameters
    ----------
    data : numpy.ndarray
        input data to be normalized
        shape: [Batch size, dimension]

    Returns
    ----------
    data : numpy.ndarry
        normalized data
        shape: [Batch size, dimension]

    References
    ----------
    .. [1] http://sebastianraschka.com/Articles/2014_about_feature_scaling.html

    '''
    numerator = data - np.min(data, 0)
    denominator = np.max(data, 0) - np.min(data, 0)
    # noise term prevents the zero division
    return numerator / (denominator + 1e-7)



timesteps = seq_length = 7
data_dim = 5
output_dim = 1

xy = np.loadtxt('data-02-stock_daily.csv', delimiter=','   )
xy = xy[::-1]
xy = MinMaxScaler(xy)

x = xy

y = xy[:, [-1]]

dataX = []
dataY = []

for i in range(0, len(y) - seq_length):
	_x = x[i:i + seq_length]
	_y = y[i + seq_length]
	print(_x, "->", _y)
	dataX.append(_x)
	dataY.append(_y)



hidden_dim = 10
train_size = int(len(dataY) * 0.7)
test_size = len(dataY) - train_size
trainX, testX = np.array(dataX[0:train_size]), np.array(dataX[train_size:len(dataX)])
trainY, testY = np.array(dataY[0:train_size]), np.array(dataY[train_size:len(dataY)])

X = tf.placeholder(tf.float32, [None, seq_length, data_dim])
Y = tf.placeholder(tf.float32, [None, 1])


cell = tf.contrib.rnn.BasicLSTMCell(num_units=hidden_dim, state_is_tuple=True)
outputs, _states = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)
Y_pred = tf.contrib.layers.fully_connected(outputs[:, -1], output_dim, activation_fn=None)



loss = tf.reduce_sum(tf.square(Y_pred - Y))


optimizer = tf.train.AdamOptimizer(0.01)
train = optimizer.minimize(loss)


sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
	_, l = sess.run([train, loss], feed_dict={X:trainX, Y:trainY})
	print(i, l)



testPredict = sess.run(Y_pred, feed_dict={X:testX})

import matplotlib.pyplot as plt
plt.plot(testY)
plt.plot(testPredict)

plt.show()








