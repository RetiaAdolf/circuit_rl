import subprocess
import os
import random
import numpy as np
import time
class Env(object):
	"""docstring for Env"""
	def __init__(self):
		super(Env, self).__init__()
		self.state = [1.0,1.0,1.0,1.0]
		self.action_space = np.array([[12, 60], [12, 60], [0.00, 0.50]])
		self.state_dim = len(self.state)

		self.eval_list = []
		for _ in range(10):
			eval_state = self.reset()
			self.eval_list.append(eval_state)

		self.state = [1.0,1.0,1.0,1.0]

	def reset(self):
		while True:
			state = [random.uniform(0, 7) for _ in range(4)]
			if sum(state) <= 10:
				self.state = state
				break
		return self.state

	def reset_eval(self, idx):
		self.state = self.eval_list[idx]
		return self.state

	def random_action(self):
		rand_aciton = []
		for action in self.action_space:
			rand_aciton.append(round(random.uniform(action[0],action[1]), 2))
		return np.array(rand_aciton)

	def step(self, actions, evaluate=False):
		M3_W, M7_W, IN_OFST = actions
		M3_W = str(M3_W)
		M7_W = str(M7_W)
		IN_OFST = str(IN_OFST)
		file_path = '../data/M3W_{}_M7W_{}_INOFST_{}.txt'.format(M3_W, M7_W, IN_OFST)
		while not os.path.exists(file_path):
			print(file_path)
			try:
				container_name = "cadence_1"
				command = 'make -C /mnt/mydata/RL/run/ M3_W={} M7_W={} IN_OFST={}'.format(M3_W, M7_W, IN_OFST)
				docker_command = ["docker", "exec", container_name, "/bin/bash", "-ic", command]
				output = subprocess.check_output(docker_command, stderr=subprocess.STDOUT)
			except:
				print("Make command timed out. Killing subprocess...")
			time.sleep(1)
		with open(file_path, 'r') as f:
			data = f.readline().split()
			while not data:
				data = f.readline().split()
			data = f.readline().split()

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
		f.close()
		#os.remove(file_path)

		outputs = [PowerDC, GBW, RmsNoise, SettlingTime]
		Weight_PowerDC = self.state[0]
		Weight_GBW = self.state[1]
		Weight_RmsNoise = self.state[2]
		Weight_SettlingTime = self.state[3]

		reward = 0

		reward += ((GBW - 1.17) / (10.098753 - 1.175)) * Weight_GBW
		if GBW < 1.17:
			reward -= 2 * Weight_GBW

		reward += ((14.7 - RmsNoise) / (14.7 - 11)) * Weight_RmsNoise
		if RmsNoise > 14.7:
			reward -= 2 * Weight_RmsNoise

		reward += ((35 - PowerDC) / (35 - 31.8)) * Weight_PowerDC
		if PowerDC > 35:
			reward -= 2 * Weight_PowerDC

		reward += ((13.5 - SettlingTime) / (13.5 - 1.72)) * Weight_SettlingTime
		if SettlingTime > 13.5:
			reward -= 2 * Weight_SettlingTime

		return outputs, reward


if __name__ == '__main__':
	test_env = Env()
	test_env.reset()
	print(test_env.state)
