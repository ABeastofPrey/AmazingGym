import tensorflow as tf
import numpy as np
import inference
import os

LOG_PATH = 'logs'
MODEL_PATH = 'models'
MODEL_NAME = 'model.ckpt'

def train():
    output = inference.inference()
    tf.summary.scalar('output', output)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        print(sess.run(output))
        tf.summary.FileWriter(LOG_PATH, sess.graph)
        tf.train.Saver().save(sess, os.path.join(MODEL_PATH, MODEL_NAME))
    
def main(argv):
    train()

if __name__ == '__main__':
    tf.app.run()