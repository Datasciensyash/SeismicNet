class Config():
	def __init__(self, config_file='config.txt'):
		self.config = eval(self.read_config_file(config_file))

	def read_config_file(self, filename):
		with open(filename, 'r') as file:
			return file.read()

	def write_config_file(self, filename):
		with open(filename, 'w') as file:
			return file.write(str(self.config))

	def __getitem__(self, x):
		return self.config[x]

	def __getattr__(self, x):
		return self.config[x]