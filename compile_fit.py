import tensorflow as tf
from tensorflow.python.keras import datasets, layers, optimizers, Sequential, metrics
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


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
network = Sequential([layers.Dense(256, activation='relu'),
                     layers.Dense(128, activation='relu'),
                     layers.Dense(64, activation='relu'),
                     layers.Dense(32, activation='relu'),
                     layers.Dense(10)])
network.build(input_shape=(None, 28*28))
network.summary()

# 定义训练用的各种参数
network.compile(optimizer=optimizers.Adam(lr=0.01),  # 优化器，用来优化参数
	        	loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # 定义loss函数，以让人知道你在优化什么
        		metrics=['accuracy']  # 定义一个存储尺用来存储东西
	            )

network.fit(db,  # 训练用哪个数据集
            epochs=5,  # 训练多少轮
            validation_data=ds_val,  # 测试用哪个数据集
            validation_freq=3  # 测试的间隔：每两个epoch就测试一次
            )
network.evaluate(ds_val)  # 最后再测试一次，因为没乱七八糟玩意，传一个测试集参数就足够了
                          # 如果不加这一行的话，最后一组不知道从哪来的又填满了一整组
                          # 加上的话就是还剩多少就测试多少


#  最后抽出一个batch来看看预测的结果究竟怎么样
sample = next(iter(ds_val))
x = sample[0]
y = sample[1] # one-hot
pred = network.predict(x) # [b, 10]  # 主要是这个函数
# convert back to number
y = tf.argmax(y, axis=1)
pred = tf.argmax(pred, axis=1)

print(pred)
print(y)
