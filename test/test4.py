import tensorflow as tf

# # tf.random.set_seed(1234)
# #  print(tf.random.uniform([2]))  # generates 'A1'
# #  print(tf.random.uniform([2]))  # generates 'A2'
#
# print(tf.random.uniform([1], seed=1))  # generates 'A1'
# print(tf.random.uniform([1], seed=1))  # generates 'A2'


tf.random.set_seed(1234)
print(tf.random.uniform([1], seed=1))  # generates 'A1'
print(tf.random.uniform([1], seed=1))  # generates 'A2'
tf.random.set_seed(1234)
print(tf.random.uniform([1], seed=1))  # generates 'A1'
print(tf.random.uniform([1], seed=1))  # generates 'A2'

