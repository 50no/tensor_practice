import tensorflow as tf
from tensorflow.python import keras

print(keras.__file__)
print(keras.datasets.__file__)

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# 获取数据集数据
(x, y), (x_test, y_test) = keras.datasets.mnist.load_data()
# 什么类型的呢 ： numpy
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.  # 这个归一化对减小损失函数特别重要
y = tf.convert_to_tensor(y, dtype=tf.int32)
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.
y_test = tf.convert_to_tensor(y_test, dtype=tf.int32)

train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
test_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
# train_iter = iter(train_db)
# sample = next(train_db)
# 手动创建超参数和步长
w1 = tf.random.truncated_normal([784, 256], stddev=0.1)
w2 = tf.random.truncated_normal([256, 128], stddev=0.1)
w3 = tf.random.truncated_normal([128, 10], stddev=0.1)
b1 = tf.zeros([256])
b2 = tf.zeros([128])
b3 = tf.zeros([10])
my_args = [w1, w2, w3, b1, b2, b3]
lr = 1e-3

for epoch in range(100):
    for step, (x, y) in enumerate(train_db):
        x = tf.reshape(x, [-1, 784])

        # 计算过程要包在tape过程中
        with tf.GradientTape() as tape:
            tape.watch(my_args)
            h1 = tf.nn.relu(x @ w1 + b1)
            h2 = tf.nn.relu(h1 @ w2 + b2)
            out = h2 @ w3 + b3

            y_onehot = tf.one_hot(y, depth=10)
            loss = tf.losses.MSE(y_onehot, out)
            loss = tf.reduce_mean(loss)

        # 通过tape计算梯度
        grads = tape.gradient(loss, my_args)
        w1.assign_sub = (lr * grads[0])
        w2.assign_sub = (lr * grads[1])
        w3.assign_sub = (lr * grads[2])
        b1.assign_sub = (lr * grads[3])
        b2.assign_sub = (lr * grads[4])
        b3.assign_sub = (lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss: ', float(loss))


    total_correct, total_num = 0, 0
    for step, (x, y) in enumerate(test_db):
        x = tf.reshape(x, [-1, 784])
        h1 = tf.nn.relu(x @ w1 + b1)
        h2 = tf.nn.relu(h1 @ w2 + b2)
        out = h2 @ w3 + b3
        prob = tf.nn.softmax(out, axis=1)
        print('prob.shape: ',end='')
        print(prob.shape)
        pred = tf.argmax(prob, axis=1)
        pred = tf.cast(pred, dtype=tf.int32)
        correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)