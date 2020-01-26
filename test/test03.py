import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)
    x = tf.reshape(x, [28*28])
    # y = tf.one_hot(y, depth=10)
    y = tf.cast(y, dtype=tf.int32)

    return x, y

BATCH_SIZE = 128
(x, y), (x_test, y_test) = datasets.mnist.load_data()
# print(x.shape)
# print(y.shape)
db = tf.data.Dataset.from_tensor_slices((x, y))
db = db.map(preprocess).shuffle(10000).batch(BATCH_SIZE)
db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
db_test = db_test.map(preprocess).batch(BATCH_SIZE)

print(next(iter(db)))

model = Sequential([
    layers.Dense(256, activation=tf.nn.relu),
    layers.Dense(128, activation=tf.nn.relu),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10, activation=tf.nn.relu),
])

model.build(input_shape=[None, 28*28])
model.summary()
optimizer = optimizers.Adam(lr=1e-3)

def main():
    for epoch in range(50):
        for step, (x, y) in enumerate(db):
            with tf.GradientTape() as tape:
                logits = model(x)
                y = tf.one_hot(y, depth=10)
                loss_ce = tf.losses.categorical_crossentropy(y, logits, from_logits=True)
                loss_ce = tf.reduce_mean(loss_ce)
            grads = tape.gradient(loss_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

        total_correct = 0
        total_num = 0
        for x, y in db_test:
            logits = model(x)
            prob = tf.nn.softmax(logits, axis=1)
            pred = tf.argmax(prob, axis=1)
            pred = tf.cast(pred, dtype=tf.int32)

            # print(pred)
            # print(y)
            correct = tf.cast(tf.equal(pred, y) ,dtype=tf.int32)
            correct = tf.reduce_sum(correct)
            # print(int(correct))


            total_correct += int(correct)
            total_num += x.shape[0]
            # print(total_num)

        acc = total_correct / total_num
        print(epoch, 'test acc: ', acc)

if __name__ == '__main__':
    main()