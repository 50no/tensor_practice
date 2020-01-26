import tensorflow as tf
from tensorflow.keras import layers, optimizers, datasets, Sequential
import os
from resnet import resnet18

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
tf.random.set_seed(2345)





def preprocess(x, y):
    # [-1~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=100)
    return x,y


(x,y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)


train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(1000).map(preprocess).batch(128)

test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(128)

# sample = next(iter(train_db))
# print('sample:', sample[0].shape, sample[1].shape,
#       tf.reduce_min(sample[0]), tf.reduce_max(sample[0]))


while True:
    network = resnet18()

    # 这句话没有什么卵用啊
    network.build(input_shape=(None, 32, 32, 3))

    network.compile(optimizer=optimizers.Adam(lr=0.01),  # 优化器，用来优化参数
                    loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # 定义loss函数，以让人知道你在优化什么
                    metrics=['accuracy']  # 定义一个存储尺用来存储东西
                    )


    # 创建完成网络后加载权重
    try:
        print('加载权重中...')
        network.load_weights('weights.ckpt')
    except Exception as e:
        print('加载失败！重新训练！')
        network.summary()

    network.fit(train_db,  # 训练用哪个数据集，！！！！！
                # 这里db的标签需要提前进行onehot编码
                # x也是需要提前reshape的
                epochs=3,  # 训练多少轮
                validation_data=test_db,  # 测试用哪个数据集
                validation_freq=3  # 测试的间隔：每n个epoch就测试一次
                )
    network.evaluate(test_db)

    try:
        print('保存权重中...')
        network.save_weights('weights.ckpt')
    except Exception as e:
        with open('./mylog.txt', 'a') as file_object:
            file_object.write(str(e))

# 一轮1.4个小时
# os.system('shutdown -s -f -t 20')
# def main():
#
#     # [b, 32, 32, 3] => [b, 1, 1, 512]
#     model = resnet18()
#     model.build(input_shape=(None, 32, 32, 3))
#     model.summary()
#     optimizer = optimizers.Adam(lr=1e-3)
#
#     for epoch in range(500):
#
#         for step, (x,y) in enumerate(train_db):
#
#             with tf.GradientTape() as tape:
#                 # [b, 32, 32, 3] => [b, 100]
#                 logits = model(x)
#                 # [b] => [b, 100]
#                 y_onehot = tf.one_hot(y, depth=100)
#                 # compute loss
#                 loss = tf.losses.categorical_crossentropy(y_onehot, logits, from_logits=True)
#                 loss = tf.reduce_mean(loss)
#
#             grads = tape.gradient(loss, model.trainable_variables)
#             optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#             if step %50 == 0:
#                 print(epoch, step, 'loss:', float(loss))
#
#
#
#         total_num = 0
#         total_correct = 0
#         for x,y in test_db:
#
#             logits = model(x)
#             prob = tf.nn.softmax(logits, axis=1)
#             pred = tf.argmax(prob, axis=1)
#             pred = tf.cast(pred, dtype=tf.int32)
#
#             correct = tf.cast(tf.equal(pred, y), dtype=tf.int32)
#             correct = tf.reduce_sum(correct)
#
#             total_num += x.shape[0]
#             total_correct += int(correct)
#
#         acc = total_correct / total_num
#         print(epoch, 'acc:', acc)
#
#
#
# if __name__ == '__main__':
#     main()