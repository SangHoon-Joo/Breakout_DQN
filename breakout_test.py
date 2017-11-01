##
## I've got a lot of help from a Chan-uk Joo's code and Sung Kim's code
## (https://github.com/jcwleo/Reinforcement_Learning/tree/master/Breakout)
## (https://github.com/hunkim/ReinforcementZeroToAll/blob/master/07_3_dqn_2015_cartpole.py)
## Thanks a lot to above two people!!
##


import gym
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import random
from scipy.misc import imresize
from skimage.color import rgb2gray
from skimage.transform import resize
from collections import deque


tf.set_random_seed(777)

learning_rate = 0.005
dis = .99
MAX_EPISODE = 500000
BATCH_SIZE = 28
REPLAY_MEMORY = 50000

history = np.zeros([42, 42, 4], dtype=np.uint8)

env = gym.make('Breakout-v0')

output_size = 3

def crop_image(image, height_range=(34,195)):
	h_begin, h_end = height_range
	return image[h_begin:h_end, ...]

def resize_image(image, HW_range):
	return imresize(image, HW_range, interp="nearest")

def make_gray_image(image):
	return rgb2gray(image)

def pre_proc(image):
	temp_image = crop_image(image)

	temp_image = make_gray_image(temp_image)

	final_image = resize_image(temp_image, (42, 42))

	return final_image

def init_history(state):
	for i in range(4):
		history[:, :, i] = state

def add_to_history(new_state):
	for i in range(3):
		history[:, :,i] = history[:, :, i+1]
	history[:, :, -1] = new_state

def get_next_history(new_state):
	next_history = history
	for i in range(3):
		next_history[:, :, i] = next_history[:, :, i+1]
	next_history[:, :, -1] = new_state
	return next_history


def reshape_history(history):
	return np.reshape(history, [-1, 42, 42, 4])

class DQN:
	def __init__(self, session, name):
		self.session = session
		self.input_size = [42,42,4]
		self.output_size = 3
		self.name = name

		self._build_network()
	
	def _build_network(self):
		with tf.variable_scope(self.name):
			self.X = tf.placeholder(shape=[None,42,42,4], dtype=tf.float32,name="input_x")

			W1 = tf.Variable(tf.random_normal([4,4,4,32], stddev=0.01))
			L1 = tf.nn.conv2d(self.X, W1, strides=[1,2,2,1], padding='VALID')
			L1 = tf.nn.relu(L1)
			L1 = tf.nn.max_pool(L1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

			L1_flat = tf.reshape(L1, [-1,10*10*32])

			W2 = tf.get_variable("W2", shape=[10*10*32, 318],
					initializer=tf.contrib.layers.xavier_initializer())
			b2 = tf.Variable(tf.random_normal([318]))
			L2 = tf.nn.relu(tf.matmul(L1_flat, W2) + b2)

			W3 = tf.get_variable("W3", shape=[318, 3],
					initializer=tf.contrib.layers.xavier_initializer())
			self.Qpred = tf.matmul(L2, W3)
			self.Y = tf.placeholder(shape=[None, output_size], dtype=tf.float32)

			self.loss = tf.reduce_sum(tf.square(self.Y-self.Qpred))
			self.train = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(self.loss)

	def predict(self, history):
		return self.session.run(self.Qpred, feed_dict={self.X: reshape_history(history)})

	def update(self, x_stack, y_stack):
		feed = {
			self.X: x_stack,
			self.Y: y_stack
		}
		return self.session.run([self.loss, self.train], feed)

def get_copy_var_ops(*, dest_scope_name="target", src_scope_name="main"):
	op_holder = []

	src_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
		scope=src_scope_name)
	dest_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES,
		scope=dest_scope_name)

	for src_var, dest_var in zip(src_vars, dest_vars):
		op_holder.append(dest_var.assign(src_var.value()))

	return op_holder


def simple_replay_train(mainDQN, targetDQN, train_batch):
	state_array = np.array([x[0] for x in train_batch])
	action_array = np.array([x[1] for x in train_batch])
	reward_array = np.array([x[2] for x in train_batch])
	next_state_array = np.array([x[3] for x in train_batch])
	done_array = np.array([x[4] for x in train_batch])

	X_batch = state_array

	Y_batch = mainDQN.predict(state_array)
	
	for i in range(len(train_batch)):
		if done_array[i] == True:
			Y_batch[i, action_array[i]] = reward_array[i]
		else:
			Y_batch[i, action_array[i]] = reward_array[i] + dis*np.max(targetDQN.predict(next_state_array[i]))
	
	loss, _ = mainDQN.update(X_batch, Y_batch)

	return loss

def main():
	replay_buffer = deque(maxlen=REPLAY_MEMORY)
	last_100_game_reward = deque(maxlen=100)

	with tf.Session() as sess:
		mainDQN = DQN(sess, 'main')
		targetDQN = DQN(sess, 'target')

		sess.run(tf.global_variables_initializer())

		copy_ops = get_copy_var_ops(dest_scope_name="target",
									src_scope_name="main")

		sess.run(copy_ops)
		
		for i in range(MAX_EPISODE):
			e = 1./ ((i/500)+1.)
			done = False
			state = env.reset()
			x = pre_proc(state)
			init_history(x)

			step_count = 0
			reward_sum = 0

			while not done:
				if np.random.rand(1) < e:
					action = np.random.randint(3)
				else:
					action = np.argmax(mainDQN.predict(history))

				next_state, reward, done, info = env.step(action+1)

				if info['ale.lives'] < 5:
					done = True

				if done:
					reward = -100
				
				next_x = pre_proc(next_state)
				next_history = get_next_history(next_x)

				replay_buffer.append((history, action, reward, next_history, done))
				if len(replay_buffer) > REPLAY_MEMORY:
					replay_buffer.popleft()

				state = next_state
				x = pre_proc(state)
				add_to_history(x)

				step_count += 1
				if i % 20 == 0 and i > 50000:
					env.render()


			if step_count > 10000:
				pass
				break

			if i % 20 == 0 and i != 0 and i != 10:
				for j in range(50):
					minibatch = random.sample(replay_buffer, 10)
					loss = simple_replay_train(mainDQN, targetDQN, minibatch)
				if i % 1000 == 0:
					print("Episode: {}, Loss: {}".format(i, loss))

			if i % 20 == 0:
				sess.run(copy_ops)

		bot_play(mainDQN)

def bot_play(mainDQN, env = env):	
	state = env.reset()
	reward_sum = 0
	x = pre_proc(state)
	init_history(x)
	env.step(1)
	while True:
		env.render()

		Qs = mainDQN.predict(history)
		action = np.argmax(Qs)

		state, reward, done, info = env.step(action+1)
		reward_sum += reward
		x = pre_proc(state)
		add_to_history(x)

		if info['ale.lives'] < 5:
			done = True

		if done:
			print("Total score: {}".format(reward_sum))
			break

if __name__ == "__main__":
	main()
