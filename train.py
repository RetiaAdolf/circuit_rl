from Env import Env
from Agent import SAC
import numpy as np
import random

TRAIN_ITER = 1000000
EVAL_INTERVAL = 100
PRINT_INTERVAL = 100

eps = 1.0
start_iters = 0
model_path = "checkpoints/sac_checkpoint_EDA_iter_{}".format(start_iters)
log = open('train_log.txt', 'a')

qf1_loss_log = []
qf2_loss_log = []
policy_loss_log = []
eval_reward_log = []

SimEnv = Env()
agent = SAC(input_dim=SimEnv.state_dim, action_space=SimEnv.action_space, hidden_dim=64)
if start_iters > 0:
	agent.load_checkpoint(ckpt_path=model_path, evaluate=True)
for i in range(start_iters + 1, TRAIN_ITER + 1):
	state = SimEnv.reset()
	if random.random() < eps:
		action = SimEnv.random_action()
	else:
		action = agent.select_action(state, evaluate=False)
	output, reward = SimEnv.step(action)
	if i % PRINT_INTERVAL == 0:
		print("current iter {}, weight {}, action {}, reward {}, output {}".format(i, state, action, reward, output))
	agent.store([state, action, reward])
	qf1_loss, qf2_loss, policy_loss = agent.learn()
	qf1_loss_log.append(qf1_loss)
	qf2_loss_log.append(qf2_loss)
	policy_loss_log.append(policy_loss)
	eps = max(0.05, eps * 0.999)

	if i % EVAL_INTERVAL == 0:
		agent.save_checkpoint(env_name="EDA", suffix="iter_{}".format(i))
		for idx in range(len(SimEnv.eval_list)):
			state = SimEnv.reset_eval(idx)
			action = agent.select_action(state, evaluate=True)
			_, reward = SimEnv.step(action)
			eval_reward_log.append(reward)
		qf1_loss_log = np.array(qf1_loss_log)
		qf2_loss_log = np.array(qf2_loss_log)
		policy_loss_log = np.array(policy_loss_log)
		eval_reward_log = np.array(eval_reward_log)
		print("current iter {}, qf1 mean loss {}, qf2 mean loss {}, policy mean loss {}, eval mean reward {}".format(
			i,
			np.mean(qf1_loss_log),
			np.mean(qf2_loss_log),
			np.mean(policy_loss_log),
			np.mean(eval_reward_log),))
		log.write("current iter {}, qf1 mean loss {}, qf2 mean loss {}, policy mean loss {}, eval mean reward {}".format(
			i,
			np.mean(qf1_loss_log),
			np.mean(qf2_loss_log),
			np.mean(policy_loss_log),
			np.mean(eval_reward_log),))
		log.write('\n')
		log.flush()
		qf1_loss_log = []
		qf2_loss_log = []
		policy_loss_log = []
		eval_reward_log = []