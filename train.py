from trainrunner import TrainRunner 
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config')
	parser.add_argument('--path')
	args = parser.parse_args()

	config = args.config
	path = args.path

	TrainRunner(config).train(path)
