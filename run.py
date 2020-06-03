import gym
import time

envs = ["starpilot", "coinrun", "bossfight", "bigfish", "caveflyer", "chaser", "climber", "heist",
 "jumper", "leaper", "maze", "miner", "ninja", "plunder", "fruitbot", "dodgeball"]

"""
env = gym.make("procgen:procgen-{}-v0".format("miner"))
obs = env.reset()
print(obs.shape)
env.render(mode='human')
"""

for e in envs:
	env = gym.make("procgen:procgen-{}-v0".format(e))
	obs = env.reset()
	env.render(mode='human')

	for i in range(100):
	    action = env.action_space.sample()
	    _, _, done, _ = env.step(action)
	    env.render(mode='human')
	    time.sleep(0.05)
	    if done:
	    	env.reset()

	env.close()