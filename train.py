import os
import retro
import inference
import numpy as np
import tensorflow as tf

# env = retro.make(game='Airstriker-Genesis', state='Level1')

LOG_PATH = 'logs'
MODEL_PATH = 'models'
MODEL_NAME = 'model.ckpt'

# IMAGE_CHANNELS = env.observation_space.shape[2] # 3
# IMAGE_HEIGHT = env.observation_space.shape[0] # 224
# IMAGE_WIDTH = env.observation_space.shape[1] # 320

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 320
IMAGE_CHANNELS = 3
ACTION_SPACE = 12
BATCH_SIZE = 32
TRAINING_STEPS = 50

def train():
    observations = tf.placeholder(name='observations', shape=[1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float32)
    actions = tf.placeholder(name='actions', shape=[None, ACTION_SPACE], dtype=tf.float32)
    values = tf.placeholder(name='values', shape=[None], dtype=tf.float32)

    q_value = inference.deep_network(observations)
    loss = inference.loss_op(q_value, actions, values)
    train_op = inference.train_op(loss)
    
    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        fake_values = np.random.random([BATCH_SIZE])
        for i in range(TRAINING_STEPS):
            # fake data
            fake_observations = np.random.random((1, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
            fake_actions = np.random.random([BATCH_SIZE, ACTION_SPACE])

            _, result, rs = sess.run([train_op, loss, merged], feed_dict={
                observations: fake_observations,
                actions: fake_actions,
                values: fake_values
            })
            writer.add_summary(rs, i)
            print(i, result)
        # tf.train.Saver().save(sess, os.path.join(MODEL_PATH, MODEL_NAME))
    
def main(argv):
    train()

if __name__ == '__main__':
    tf.app.run()