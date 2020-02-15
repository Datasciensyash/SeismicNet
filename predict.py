from trainrunner import TrainRunner 
import argparse

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--config')
	parser.add_argument('--path')
	parser.add_argument('-filename')		
	args = parser.parse_args()

	config = args.config
	data_path = args.path
	filename = args.filename

	TrainRunner(config).predict(data_path, filename)
