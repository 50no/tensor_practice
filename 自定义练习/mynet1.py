import  tensorflow as tf
from    tensorflow.python.keras import datasets, layers, optimizers, Sequential, metrics
from 	tensorflow import keras
import  os

class MyCnnLayer(layers.Layer):
    def __init__(self):
        super(MyCnnLayer, self).__init__()
        self.conv1 = layers.Conv2D(512, (3, 3), strides=2, padding='same')
        self.bn1 = layers.BatchNormalization()
        self.relu = layers.Activation('relu')

        self.conv2 = layers.Conv2D(512, (3, 3), strides=1, padding='same')
        self.bn1 = layers.BatchNormalization()


    def call(self, inputs, training=None):
        out = self.conv1(inputs)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn1(out)

        return out


class MyRnnLayer(layers.Layer):
    def __init__(self):
        super(MyRnnLayer, self).__init__()

        self.state0 = [tf.zeros([128, 128])]
        self.state1 = [tf.zeros([128, 128])]

        self.rnn_cell0 = layers.SimpleRNNCell(128, dropout=0.5)
        self.rnn_cell1 = layers.SimpleRNNCell(128, dropout=0.5)


    def call(self, inputs, training=None):
        print(inputs.shape)
        x = tf.reshape(inputs, [512, 196])

        state0 = self.state0
        state1 = self.state1
        for word in tf.unstack(x, axis=1):
            out0, state0 = self.rnn_cell0(word, state0, training)
            out1, state1 = self.rnn_cell0(out0, state1, training)

        print(out1.shape)
        return out1


class MyDense(layers.Layer):
    def __init__(self):
        super(MyDense, self).__init__()

        self.mylayer = Sequential([layers.Dense(64, activation='relu'),
                                   layers.Dense(32, activation='relu'),
                                   layers.Dense(16, activation='relu'),
                                   layers.Dense(10, activation='relu'),])


    def call(self, inputs):
        out = self.mylayer(inputs)
        return out


class MyNetwork(keras.Model):
    def __init__(self, ):
        super(MyNetwork, self).__init__()

        self.cnn = MyCnnLayer()
        self.rnn = MyRnnLayer()
        self.dense = MyDense()

    def call(self, inputs):
        out = self.cnn(inputs)
        out = self.rnn(out)
        out = self.dense(out)

        return out


def new_net(inputs):
    return MyNetwork(inputs)

input = tf.random.normal([28, 28])
print(new_net(input))



