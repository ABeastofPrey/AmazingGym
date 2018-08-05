import retro
import tensorflow as tf

env = retro.make(game='Airstriker-Genesis', state='Level1')
features = env.observation_space.shape[0]*env.observation_space.shape[1]*env.observation_space.shape[2]

def test_env():
    observation = env.reset()
    # print(env.observation_space)
    # print(env.observation_space.shape)
    # print(env.observation_space.high)
    # print(env.observation_space.high.shape)
    # print(env.observation_space.low)
# Box(224, 320, 3)
# (224, 320, 3)
    while True:
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            env.reset()
        env.render()

def main(argv):
    test_env()

# import train
# import inference
# import tensorflow as tf

# def eval():
#     with tf.Graph().as_default() as g:
#         output = inference.inference()
#         with tf.Session() as sess:
#             # tf.train.get_checkpoint_state函数会通过checkpoint文件自动找到目录中最新模型的文件名。
#             ckpt = tf.train.get_checkpoint_state(train.MODEL_PATH)
#             if ckpt and ckpt.model_checkpoint_path:
#                 tf.train.Saver().restore(sess, ckpt.model_checkpoint_path)
#                 print(sess.run(output))

# def main(argv):
#     eval()
    
if __name__ == '__main__':
    tf.app.run()