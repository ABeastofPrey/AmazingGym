import tensorflow as tf
import numpy as np


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

## prepare the original data
with tf.name_scope('data'):
     x_data = np.random.rand(100).astype(np.float32)
     y_data = 0.3*x_data+0.1
##creat parameters
with tf.name_scope('parameters'):
     with tf.name_scope('weights'):
         weight = tf.Variable(tf.random_uniform([1],-1.0,1.0))
         variable_summaries(weight)
     with tf.name_scope('biases'):
         bias = tf.Variable(tf.zeros([1]))
         variable_summaries(bias)
##get y_prediction
with tf.name_scope('y_prediction'):
     y_prediction = weight*x_data+bias
##compute the loss
with tf.name_scope('loss'):
     loss = tf.reduce_mean(tf.square(y_data-y_prediction))
     tf.summary.scalar('loss',loss)
#creat train ,minimize the loss 
with tf.name_scope('train'):
     train = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
#creat init
with tf.name_scope('init'): 
     init = tf.global_variables_initializer()
##creat a Session 
sess = tf.Session()
#merged
merged = tf.summary.merge_all()
##initialize
writer = tf.summary.FileWriter("logs/", sess.graph)
sess.run(init)
## Loop
for step  in  range(101):
    sess.run(train)
    rs=sess.run(merged)
    writer.add_summary(rs, step)
