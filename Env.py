import subprocess
import signal
import os
import random
import numpy as np
import time
class Env(object):
	"""docstring for Env"""
	def __init__(self):
		super(Env, self).__init__()
		self.state = np.ones(4)
		self.action_space = np.array([[12, 60], [12, 60], [0.00, 0.50]])
		self.action_step = np.array([[-5, 5], [-5, 5], [-0.05, 0.05]])
		self.state_dim = len(self.state)

		self.eval_list = []
		for _ in range(10):
			eval_state = self.reset()
			self.eval_list.append(eval_state)

		self.state = np.ones(4)

	def __softmax__(self, weight):
		exp_weight = np.exp(weight / 0.2) 
		return exp_weight / exp_weight.sum(axis=-1)

	def reset(self):
		rand_state = np.random.rand(4)
		self.state = (self.__softmax__(rand_state).round(2) * 10) + 1
		return self.state

	def reset_eval(self, idx):
		self.state = self.eval_list[idx]
		return self.state

	def optimize_action(self, state, base_action):
		tasks = []
		actions = []
		files = []
		add_base = False
		while len(tasks) < 8:
			if not add_base:
				action = base_action
				add_base = True
			else:
				action = self.random_action_with_base(state, base_action)
			M3_W, M7_W, IN_OFST = action
			M3_W = str(M3_W)
			M7_W = str(M7_W)
			IN_OFST = str(IN_OFST)
			file_path = '../data/M3W_{}_M7W_{}_INOFST_{}.txt'.format(M3_W, M7_W, IN_OFST)
			print(file_path)
			if not os.path.exists(file_path):
				tasks.append(action)
			actions.append(action)
			files.append(file_path)
			
		processes = []

		for i, task in enumerate(tasks):
			M3_W, M7_W, IN_OFST = task
			M3_W = str(M3_W)
			M7_W = str(M7_W)
			IN_OFST = str(IN_OFST)
			container_name = "cadence_{}".format(i+1)
			command = "make -C /mnt/mydata/RL_{}/run/ M3_W={} M7_W={} IN_OFST={}".format(i+1, M3_W, M7_W, IN_OFST)
			print("command {} at container {}".format(command, container_name))
			process = self.run_docker(container_name, command)
			processes.append(process)

		for process in processes:
			try:
				process.wait(timeout=100)
			except subprocess.TimeoutExpired:
				pgid = os.getpgid(process.pid)
				os.killpg(pgid, signal.SIGTERM)

		max_reward = -9999999
		best_action = None
		for action, file in zip(actions,files):
			data = self.__read_file__(file)
			PowerDC, GBW, RmsNoise, SettlingTime = self.__read_data__(data)
			output = [PowerDC, GBW, RmsNoise, SettlingTime]
			_normalize_output = self.__normalization__(output)
			reward = self.__get_reward__(self.state, _normalize_output)
			if reward > max_reward:
				max_reward = reward
				best_action = action
		print("optimized action {} and reward {}".format(best_action, max_reward))

		return best_action


	def random_action(self, state):
		rand_aciton = []
		for action in self.action_space:
			rand_aciton.append(round(random.uniform(action[0],action[1]), 2))
		return np.array(rand_aciton)

	def random_action_with_base(self, state, base_action):
		rand_aciton = []
		for i, step in enumerate(self.action_step):
			var = round(random.uniform(step[0],step[1]), 2)
			var_action = max(min(base_action[i] + var, self.action_space[i][1]), self.action_space[i][0])
			rand_aciton.append(round(var_action,2))
		return np.array(rand_aciton)


	def run_docker(self, container_name, command):
		#print("run command on {}".format(container_name))
		with open('/dev/null', 'w') as f:
			process = subprocess.Popen(['docker', 'exec', container_name, "/bin/bash", "-ic", command], stdout=f, stderr=f)
			#process = subprocess.Popen(['docker', 'exec', container_name, "/bin/bash", "-ic", command])
		return process

	def __read_file__(self, file_path):
		with open(file_path, 'r') as f:
			#print(file_path)
			data = f.readline().split()
			while not data:
				data = f.readline().split()
			data = f.readline().split()
		f.close()
		return data

	def __simulator_step__(self, action):
		M3_W, M7_W, IN_OFST = action
		M3_W = str(M3_W)
		M7_W = str(M7_W)
		IN_OFST = str(IN_OFST)
		file_path = '../data/M3W_{}_M7W_{}_INOFST_{}.txt'.format(M3_W, M7_W, IN_OFST)
		while not os.path.exists(file_path):
			container_name = "cadence_1"
			command = 'make -C /mnt/mydata/RL/run/ M3_W={} M7_W={} IN_OFST={}'.format(M3_W, M7_W, IN_OFST)
			process = self.run_docker(container_name, command)
			try:
				process.wait(timeout=100)
			except:
				pgid = os.getpgid(process.pid)
				os.killpg(pgid, signal.SIGTERM)

		data = self.__read_file__(file_path)
		PowerDC, GBW, RmsNoise, SettlingTime = self.__read_data__(data)

		return [PowerDC, GBW, RmsNoise, SettlingTime]

	def __read_data__(self, data):
		PowerDC = float(data[4][:-1])
		PowerDC_unit = data[4][-1]
		if PowerDC_unit == "u":
			PowerDC = PowerDC
		elif PowerDC_unit == "n":
			PowerDC = PowerDC * 1e-3
		elif PowerDC_unit == "m":
			PowerDC = PowerDC * 1e3

		GBW = float(data[5][:-1])
		GBW_unit = data[5][-1]
		if GBW_unit == "M":
			GBW = GBW
		elif GBW_unit == "K":
			GBW = GBW * 1e-3
		elif GBW_unit == "G":
			GBW = GBW * 1e3

		RmsNoise = float(data[6][:-1])
		RmsNoise_unit = data[6][-1]
		if RmsNoise_unit == "n":
			RmsNoise = RmsNoise
		elif RmsNoise_unit == "p":
			RmsNoise = RmsNoise * 1e-3
		elif RmsNoise_unit == "u":
			RmsNoise = RmsNoise * 1e3


		SettlingTime = float(data[7][:-1])
		SettlingTime_unit = data[7][-1]
		if SettlingTime_unit == "u":
			SettlingTime = SettlingTime
		elif SettlingTime_unit == "n":
			SettlingTime = SettlingTime * 1e-3
		elif SettlingTime_unit == "m":
			SettlingTime = SettlingTime * 1e3

		return PowerDC, GBW, RmsNoise, SettlingTime

	def __get_reward__(self, weight, output):
		reward = (weight * output).sum(axis=-1)
		penalty = (weight * (output < 0)).sum(axis=-1) * 0
		return reward - penalty

	def __normalization__(self, output):
		PowerDC, GBW, RmsNoise, SettlingTime = output

		PowerDC = ((35 - PowerDC) / (35 - 31.8))
		GBW = ((GBW - 1.17) / (10.098753 - 1.175))
		RmsNoise = ((14.7 - RmsNoise) / (14.7 - 11))
		SettlingTime = ((13.5 - SettlingTime) / (13.5 - 1.72))

		return np.array([PowerDC, GBW, RmsNoise, SettlingTime])

	def step(self, action, evaluate=False):

		output = self.__simulator_step__(action)
		_normalize_output = self.__normalization__(output)
		reward = self.__get_reward__(self.state, _normalize_output)

		return output, reward


if __name__ == '__main__':
	test_env = Env()
	state = test_env.reset()
	print(state)
	action = test_env.random_action(state)
	print(action)
	action = test_env.random_action_with_base(state, action)
	print(action)
	best_action = test_env.optimize_action(state, action)
	print(best_action)
