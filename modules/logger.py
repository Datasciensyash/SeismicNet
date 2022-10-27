class LogHolder():

	def __init__(self, dirpath, name):
		self.dirpath = dirpath
		self.name = name
		self.reset(suffix='train')

	def write_to_file(self):
		with open(f'{self.dirpath}{self.name}-metrics-{self.suffix}.txt', 'a') as file:
			for i in self.metric_holder:
				file.write(f'{str(i)} ')

		with open(f'{self.dirpath}{self.name}-loss-{self.suffix}.txt', 'a') as file:
			for i in self.loss_holder:
				file.write(f'{str(i)} ')		


	def init_files(self, dirpath, name):
		with open(f'{dirpath}{name}-metrics-{self.suffix}.txt', 'w') as metric_file:
			loss_file = open(f'{dirpath}{name}-loss-{self.suffix}.txt', 'w')

		loss_file.close()
		return None

	def write_metric(self, val):
		if type(val) == list:
			self.metric_holder.extend(val)
		else:
			self.metric_holder.append(val)

	def write_loss(self, val):
		self.loss_holder.append(val)

	def reset(self, suffix):
		self.metric_holder = []
		self.loss_holder = []
		self.suffix = suffix
		self.init_files(self.dirpath, self.name)