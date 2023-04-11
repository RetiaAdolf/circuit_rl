from Env import Env
from Agent import SAC
import numpy as np
import random
import sys

w_PowerDC, w_GBW, w_RmsNoise, w_SettlingTime, iters = sys.argv[1:]
model_path = "./checkpoints/sac_checkpoint_EDA_iter_{}".format(iters)

state = np.array([int(w_PowerDC), int(w_GBW), int(w_RmsNoise), int(w_SettlingTime)])
SimEnv = Env()
agent = SAC(input_dim=SimEnv.state_dim, action_space=SimEnv.action_space, hidden_dim=64)
if iters == "0":
	pass
else:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)

action = agent.select_action(state, evaluate=True)
output, reward = SimEnv.eval(state, action)
M3_W, M7_W, IN_OFST = action
M3_W = str(M3_W)
M7_W = str(M7_W)
IN_OFST = str(IN_OFST)
print("M3_W = {}, M7_W = {}, IN_OFST = {}".format(M3_W, M7_W, IN_OFST))
PowerDC, GBW, RmsNoise, SettlingTime = output
print("PowerDC = {}, GBW = {}, RmsNoise = {}, SettlingTime = {}".format(PowerDC, GBW, RmsNoise, SettlingTime))
print(reward)