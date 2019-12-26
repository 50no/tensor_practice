import tensorflow as tf
from tensorflow.python.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32) / 255.
    x = tf.reshape(x, [28*28])
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

batchsz = 128
(x, y), (x_val, y_val) = datasets.mnist.load_data()

db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(60000).batch(batchsz)
ds_val = tf.data.Dataset.from_tensor_slices((x_val, y_val))
ds_val = ds_val.map(preprocess).batch(batchsz)

# 神经网络结构肯定还是要自己建的
network = Sequential([layers.Dense(256, activation=)])