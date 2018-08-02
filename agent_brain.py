import numpy as np
import tensorflow as tf

class Brain:
    def __init__(self, actions, features, learning_rate, reward_decay, visualization=False):
        #动作空间的维数
        self.actions = actions
        #状态特征的维数
        self.features = features
        #学习速率
        self.lr = learning_rate
        #回报衰减率
        self.gamma = reward_decay
        #一条轨迹的观测值，动作值，和回报值
        self.ep_obs, self.ep_as, self.ep_rs = [],[],[]

        #创建策略网络
        self._neuron_networks()
        #启动一个默认的会话
        self.sess = tf.Session()
        if visualization:
            tf.summary.FileWriter("logs/", self.sess.graph)
        # 初始化会话中的变量
        self.sess.run(tf.global_variables_initializer())

    def _neuron_networks(self):
        # 创建输入占位符
        with tf.name_scope('input'):
            self.tf_observations = tf.placeholder(name="obs", shape=[None, self.features], dtype=tf.float32)
            self.tf_actions = tf.placeholder(name="acts", shape=[None, ], dtype=tf.int32)
            self.tf_values = tf.placeholder(name="vals", shape=[None, ], dtype=tf.float32)

        # layer1
        layer1 = tf.layers.dense(
            inputs=self.tf_observations,
            units=10, activation=tf.nn.tanh,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer1',)
    
        # layer2
        layer2 = tf.layers.dense(
            inputs=layer1,
            units=self.actions,
            activation=None,
            kernel_initializer=tf.random_normal_initializer(mean=0, stddev=0.3),
            bias_initializer=tf.constant_initializer(0.1),
            name='layer2'
        )

        # 利用softmax函数得到每个动作的概率
        self.all_actions_probability = tf.nn.softmax(layer2, name="actions_probability")

        # definition of loss function
        with tf.name_scope("loss"):
            neg_log_prob = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=layer2, labels=self.tf_actions)
            loss = tf.reduce_mean(neg_log_prob*self.tf_values)

        # definition of training operation
        with tf.name_scope("train"):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)

    def close(self):
        self.sess.close()

    def random_action(self, observation):
        probability_weight = self.sess.run(self.all_actions_probability, feed_dict={
            self.tf_observations: observation[np.newaxis,:]
        })
        action = np.random.choice(range(probability_weight.shape[1]), p=probability_weight.ravel())
        return action

    def greedy_action(self, observation):
        actions_probability = self.sess.run(self.all_actions_probability, feed_dict={
            self.tf_observations: observation[np.newaxis,:]
        })
        action = np.argmax(actions_probability.ravel())
        return action

    # 将一个回合的状态，动作和回报都保存在一起
    def store_transition(self, state, action, rward):
        self.ep_obs.append(state)
        self.ep_as.append(action)
        self.ep_rs.append(rward)

    def learn(self):
        ## 计算episode的折扣回报
        discounted_rewards = np.zeros_like(self.ep_rs)
        running_add = 0
        for t in reversed(range(0, len(self.ep_rs))):
            running_add = running_add * self.gamma + self.ep_rs[t]
            discounted_rewards[t] = running_add
        # 归一化
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)

        ## 更新参数
        self.sess.run(self.train_op, feed_dict={
            self.tf_observations: np.vstack(self.ep_obs),
            self.tf_actions: np.array(self.ep_as),
            self.tf_values: discounted_rewards
        })

        ## 清空缓存的数据
        self.ep_obs, self.ep_as, self.ep_rs = [], [],[]

        ## 返回回报
        return discounted_rewards

    def _standardize(self, X):
        """特征标准化处理
        Args:
            X: 样本集
        Returns:
            标准后的样本集
        """
        n = X.shape[1]
        # 归一化每一个特征
        for j in range(n):
            features = X[:,j]
            meanVal = features.mean(axis=0)
            std = features.std(axis=0)
            if std != 0:
                X[:, j] = (features-meanVal)/std
            else:
                X[:, j] = 0
        return X
