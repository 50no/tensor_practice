import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

BATCH_SIZE = 128

def get_data_of_mnist():
    # 返回dataset格式数据

    (x, y), (x_test, y_test) = datasets.mnist.load_data()

    def preprocess(x, y):
        x = tf.cast(x, dtype=tf.float32)
        x = tf.reshape(x, [28, 28, 1])
        y = tf.one_hot(y, depth=10)
        y = tf.cast(y, dtype=tf.int32)
        return x, y

    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_train = db_train.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
    db_test = db_test.map(preprocess).batch(BATCH_SIZE)

    return db_train, db_test


class MyNet(keras.Model):
    def __init__(self):
        super(MyNet, self).__init__()
        # 创建cnn层的玩意 [b, 28, 28, 1] => (b, 14, 14, 128)
        self.cnn1 = layers.Conv2D(64, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
        self.cnn2 = layers.Conv2D(128, (3, 3), strides=1, padding='same', activation=tf.nn.relu)
        self.maxpooling1 = layers.MaxPool2D(pool_size=[2, 2], strides=2, padding='same')
        # 创建rnn层的玩意
        self.rnn1 = layers.LSTM(64, dropout=0.5, return_sequences=True, unroll=True)
        self.rnn2 = layers.LSTM(64, dropout=0.5, unroll=True)
        # 创建最后全连接分类层的玩意
        self.fc1 = layers.Dense(32)
        self.fc2 = layers.Dense(10)

    def call(self, inputs):
        out = self.cnn1(inputs)
        out = self.cnn2(out)
        out = self.maxpooling1(out)
        out = tf.reshape(out, [-1, 14*14, 128])
        out = self.rnn1(out)
        out = self.rnn2(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

get_data_of_mnist()

# sample, _ = get_data_of_mnist()
#
# sample = next(iter(sample))[1]
# print(sample.shape)

# my_network = MyNet()
# my_network.build(input_shape=(None, 28, 28, 1))
# my_network.summary()
# out = my_network(sample)
# print(out.shape)
# #
def main():
    my_network = MyNet()
    my_network.build(input_shape=(None, 28, 28, 1))
    my_network.summary()
    my_network.compile(optimizer=keras.optimizers.Adam(1e-3),
                       loss=tf.losses.CategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy'])
    db_train, db_test = get_data_of_mnist()
    my_network.fit(db_train, epochs=5, validation_data=db_test, validation_freq=2)
    my_network.evaluate(db_test)


if __name__ == '__main__':
    main()