try:
    print('保存权重中...')
    network.save_weights('weights.ckpt')
except Exception as e:
    with open('./mylog.txt', 'a') as file_object:
        file_object.write(str(e))
print('okokokookokokok')