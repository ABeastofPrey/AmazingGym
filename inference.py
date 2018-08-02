import random
import numpy as np
import tensorflow as tf

def inference(input_tensor=[], train=True):
    input1 = tf.constant([1.0, 2.0, 3.0], name='input1')
    with tf.variable_scope('input2'):
        input2 = tf.get_variable(initializer=tf.random_uniform([3]), name='input2')
    add = tf.add_n([input1, input2], name='add')
    return add
