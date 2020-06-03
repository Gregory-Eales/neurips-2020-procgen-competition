import random



class RandomAgent(object):


	def __init__(self):
		pass


	def act(self):
		return random.randint(0, 15)



if __name__ == "__main__":

	ra = RandomAgent()


	print(ra.act())