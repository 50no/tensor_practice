import tensorflow as tf
import os
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics
from tensorflow import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

BATCH_SIZE = 128



def get_data_of_cifar10():
    """返回Dataset格式的数据"""

    def preprocess(x, y):
        x = 2 * tf.cast(x, dtype=tf.float32) / 255. - 1
        y = tf.squeeze(y)
        y = tf.one_hot(y, depth=10)
        y = tf.cast(y, dtype=tf.int32)
        return x, y

    def print1():
        print('kalsdfjl')

    (x, y), (x_test, y_test) = datasets.cifar10.load_data()
    train_db = tf.data.Dataset.from_tensor_slices((x, y))
    test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    train_db = train_db.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
    test_db = test_db.map(preprocess).batch(BATCH_SIZE)

    # sample = next(iter(train_db))
    # print(sample[1].shape)
    return train_db, test_db


class MyDense(layers.Layer):
    def __init__(self, inp_dim, out_dim):
        super(MyDense, self).__init__()

        self.kernel = self.add_variable('w', [inp_dim, out_dim])


    def __call__(self, inputs):
        x = inputs @ self.kernel
        return x


class MyNetwork(keras.Model):
    def __init__(self):
        super(MyNetwork, self).__init__()

        self.fc1 = MyDense(32*32*3, 256)
        self.fc2 = MyDense(256, 128)
        self.fc3 = MyDense(128, 64)
        self.fc4 = MyDense(64, 32)
        self.fc5 = MyDense(32, 10)


    def call(self, inputs, training=None):
        temp = tf.reshape(inputs, [-1, 32*32*3])
        temp = self.fc1(temp)
        temp = tf.nn.relu(temp)
        temp = self.fc2(temp)
        temp = tf.nn.relu(temp)
        temp = self.fc3(temp)
        temp = tf.nn.relu(temp)
        temp = self.fc4(temp)
        temp = tf.nn.relu(temp)
        output = self.fc5(temp)

        return output


my_network = MyNetwork()
# my_network.summary()
my_network.compile(optimizer=optimizers.Adam(lr=1e-3),
                   loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy'])
train_db, test_db = get_data_of_cifar10()
my_network.fit(train_db, epochs=15, validation_data=test_db, validation_freq=1)
my_network.evaluate(test_db)