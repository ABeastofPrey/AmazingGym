import train
import random
import numpy as np
import tensorflow as tf
from collections import deque

# 第一层卷积层的尺寸和深度。
CONV1_SIZE = 8
CONV1_DEEP = 32

# 第二层卷积层的尺寸和深度。
CONV2_SIZE = 4
CONV2_DEEP = 64

# 第三层卷积层的尺寸和深度。
CONV3_SIZE = 3
CONV3_DEEP = 64

# 全连接节点的个数
FC1_SIZE = 512

EPSILON = 0.8

# Store transition memory
Memory = deque()
MEMORY_SIZE = 500

GAMMA = 0.99
LEARN_RATE = 0.01

def deep_network(observation):
    with tf.variable_scope('layer1_conv1', reuse=False):
        filter = tf.get_variable(name='filter', shape=[CONV1_SIZE, CONV1_SIZE, train.IMAGE_CHANNELS, CONV1_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[CONV1_DEEP], initializer=tf.constant_initializer(0.0))
        # variable_summaries(filter)
        # variable_summaries(biases)
        conv = tf.nn.conv2d(name='conv', input=observation, filter=filter, strides=[1, 4, 4, 1], padding='SAME')
        relu1 = tf.nn.relu(tf.nn.bias_add(conv, biases))

    with tf.variable_scope('layer2_conv2', reuse=False):
        filter = tf.get_variable(name='filter', shape=[CONV2_SIZE, CONV2_SIZE, CONV1_DEEP, CONV2_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[CONV2_DEEP], initializer=tf.constant_initializer(0.0))
        # variable_summaries(filter)
        # variable_summaries(biases)
        conv = tf.nn.conv2d(name='conv', input=relu1, filter=filter, strides=[1, 2, 2, 1], padding="SAME")
        relu2 = tf.nn.relu(tf.nn.bias_add(conv, biases))

    with tf.variable_scope('layer3_conv3', reuse=False):
        filter = tf.get_variable(name='filter', shape=[CONV3_SIZE, CONV3_SIZE, CONV2_DEEP, CONV3_DEEP], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[CONV3_DEEP], initializer=tf.constant_initializer(0.0))
        # variable_summaries(filter)
        # variable_summaries(biases)
        tf.summary.histogram('bias', biases)
        conv = tf.nn.conv2d(name='conv', input=relu2, filter=filter, strides=[1, 1, 1, 1], padding='SAME')
        relu3 = tf.nn.relu(tf.nn.bias_add(conv, biases))
            
    with tf.name_scope('reshape_op'):
        shape = relu3.get_shape().as_list() # [1, 28, 40, 64]
        nodes = shape[1] * shape[2] * shape[3] # 71680
        reshaped = tf.reshape(relu3, [-1, nodes]) # Tensor("Reshape:0", shape=(1, 71680), dtype=float32)
    
    with tf.variable_scope('layer4_fc1', reuse=False):
        weights = tf.get_variable(name='weights', shape=[nodes, FC1_SIZE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biases', shape=[FC1_SIZE], initializer=tf.constant_initializer(0.0))
        # variable_summaries(weights)
        # variable_summaries(biases)
        fc1 = tf.nn.relu(tf.matmul(reshaped, weights) + biases)

    with tf.variable_scope('layer5_fc2'):
        weights = tf.get_variable(name='weights', shape=[FC1_SIZE, train.ACTION_SPACE], initializer=tf.truncated_normal_initializer(stddev=0.1))
        biases = tf.get_variable(name='biase', shape=[train.ACTION_SPACE], initializer=tf.constant_initializer(0.0))
        with tf.name_scope('weights'):
            variable_summaries(weights)
        with tf.name_scope('biases'):
            variable_summaries(biases)
        logit = tf.matmul(fc1, weights) + biases # q_value
    return logit

def loss_op(logit, actions, q_target):
    with tf.name_scope('loss'):
        # q_action = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logit, labels=actions))
        q_action = tf.reduce_sum(tf.multiply(logit, actions), 1)
        loss = tf.reduce_mean(tf.square(q_target - q_action))
        tf.summary.scalar('loss',loss)
    return loss

def train_op(loss):
    with tf.name_scope('train_op'):
        train = tf.train.AdamOptimizer(LEARN_RATE).minimize(loss)
    return train

def epsilon_action(logit, cur_observation, observations):
    action = np.zeros(train.ACTION_SPACE)
    if random.random() < EPSILON:
        q_value = logit.eval(feed_dict={observations: cur_observation[np.newaxis,:]})
        index = np.argmax(q_value[0])
    else:
        index = random.randrange(train.ACTION_SPACE)
    action[index] = 1
    return action

def store_transition(observation, action, reward, next_observation, terminal):
    Memory.append((observation, action, reward, next_observation, terminal))
    if len(Memory) > MEMORY_SIZE: Memory.popleft()

def learn(train_op, logit, observations, actions, q_target):
    batch_count = train.BATCH_SIZE
    if train.BATCH_SIZE > len(Memory):
        batch_count = len(Memory)
    batch_data = random.sample(Memory, batch_count)
    cu_obs_batch = [data[0] for data in batch_data]
    action_batch = [data[1] for data in batch_data]
    reward_batch = [data[2] for data in batch_data]
    ne_obs_batch = [data[3] for data in batch_data]

    t_value_batch = []
    q_value_batch = logit.eval(feed_dict={observations: ne_obs_batch})
    for i in range(batch_count):
        terminal = batch_data[i][4]
        if terminal:
            t_value_batch.append(reward_batch[i])
        else:
            t_value_batch.append(reward_batch[i] + GAMMA * np.max(q_value_batch[i]))

    train_op.run(feed_dict={
        observations: cu_obs_batch,
        actions: action_batch,
        q_target: t_value_batch
    })

def variable_summaries(var):
    """对一个张量添加多个描述。
    
    Arguments:
        var {[Tensor]} -- 张量
    """
    
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean) # 均值
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev) # 标准差
        tf.summary.scalar('max', tf.reduce_max(var)) # 最大值
        tf.summary.scalar('min', tf.reduce_min(var)) # 最小值
        tf.summary.histogram('histogram', var)
