import os
import retro
import random
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
TRAINING_STEPS = 15

def test():
    observations = tf.placeholder(name='observations', shape=[None, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS], dtype=tf.float32)
    actions = tf.placeholder(name='actions', shape=[None, ACTION_SPACE], dtype=tf.float32)
    q_target = tf.placeholder(name='q_target', shape=[None], dtype=tf.float32)

    logit = inference.deep_network(observations)
    
    loss = inference.loss_op(logit, actions, q_target)
    train_op = inference.train_op(loss)

    with tf.Session() as sess:
        tf.global_variables_initializer().run()
        merged = tf.summary.merge_all()
        writer = tf.summary.FileWriter(LOG_PATH, sess.graph)

        for i in range(TRAINING_STEPS):
            # # fake data
            # fake_observations = np.random.random((BATCH_SIZE, IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
            # fake_actions = np.random.random([BATCH_SIZE, ACTION_SPACE])
            # fake_values = np.random.random([BATCH_SIZE])
            # _, result, rs = sess.run([train_op, loss, merged], feed_dict={
            #     observations: fake_observations,
            #     actions: fake_actions,
            #     q_target: fake_values
            # })
            # writer.add_summary(rs, i)
            # print(i, result)

            observation = np.random.random((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
            observation_ = np.random.random((IMAGE_HEIGHT, IMAGE_WIDTH, IMAGE_CHANNELS)) 
            action = inference.epsilon_action(logit, observation_, observations)
            done = random.choice([True, False])
            reward = random.random()
            inference.store_transition(observation, action, reward, observation_, done)
        inference.learn(train_op, logit, observations, actions, q_target)
        # tf.train.Saver().save(sess, os.path.join(MODEL_PATH, MODEL_NAME))

# def train():
#     step = 0
#     for episode in range(TRAINING_STEPS):
#         observation = env.reset()
#         while True:
#             action = inference.epsilon_action(observation)
#             next_observation, reward, done, info = env.step(action)
#             inference.store_transition(observation, action, reward, next_observation)
#             if (step > 200) and (step % 5 == 0):
#                 inference.learn()
#             observation = next_observation
#             if done: break
#             step += 1
#     env.close()
    
def main(argv):
    test()

if __name__ == '__main__':
    tf.app.run()