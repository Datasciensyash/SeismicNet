from trainrunner import TrainRunner 
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config')
	args = parser.parse_args()

	config = args.config

	TrainRunner(config).train()
