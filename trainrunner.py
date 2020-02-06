from modules.metrics import iou_pytorch
from modules.losses import ce_Loss, jaccard_loss, dice_loss

from modules.config import Configurator
from modules.logger import LogHolder

from modules.dataset import SeismicDataset
from modules.augmentations import VerticalShift, HorizontalShift, InvertImg

from model.unet import UNet

import albumentations
import numpy as np
import torch

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class TrainRunner():
	def __init__(self, config_file='config.txt'):

		CONFIGURATION = Config(config_file)
		LOGGER = LogHolder(CONFIGURATION.LOGDIR, CONFIGURATION.LOGNAME)

		#Initialization of model
		self.model = self.init_model(CONFIGURATION.CHANNELS_IN, CONFIGURATION.CHANNELS_OUT, CONFIGURATION.LOAD_MODEL, CONFIGURATION.MODEL_LOAD_PATH, CONFIGURATION.MODEL_NAME)

		#Initialization of device
		self.device = self.init_device(CONFIGURATION.DEVICE)

		#Initialization of optimizer
		self.optimizer = self.init_optimizer(CONFIGURATION.ADAM_LR, self.model.parameters())

		#Initialization of loss function
		self.criterion = self.init_criterion(CONFIGURATION.CLASS_WEIGHT_0, CONFIGURATION.CLASS_WEIGHT_1, self.device)

		#Initialization of metric function
		self.metric = self.init_metric()

		#Initialization of augmentation function
		self.aug = self.init_augmentation(CONFIGURATION)

		self.CONFIGURATION = CONFIGURATION
		self.LOGGER = LOGGER
	

	def init_model(self, CHANNELS_IN, CHANNELS_OUT, LOAD_MODEL, MODEL_LOAD_PATH, MODEL_NAME):
		"""
		Initialization and loading model if needed.
			Int: CHANNELS_IN -> Number of input channels in UNet
			Int: CHANNELS_OUT -> Number of output channels in UNet
			Bool: LOAD_MODEL -> If True we need to load existing parameters.
			Str: MODEL_LOAD_PATH -> Path where models are stored
			Str: MODEL_NAME -> Name of loading model

		Returns: Model
		"""

		model = UNet(CHANNELS_IN, CHANNELS_OUT)
		if LOAD_MODEL:
			model_state_dict = torch.load(MODEL_LOAD_PATH + MODEL_NAME)
			model.load_state_dict(model_state_dict)

		return model

	def init_optimizer(self, ADAM_LR, parameters):
		"""
		Initialization of optimizer.
			Float: ADAM_LR -> Learning rate for Adam
			torch.Tensor: parameters -> Parameters to optimize

		Returns: Optimizer
		"""
		optimizer = torch.optim.Adam(parameters, lr=ADAM_LR)
		return optimizer

	def init_criterion(self, CLASS_WEIGHT_0, CLASS_WEIGHT_1, device):
		"""
		Initialization of loss function:
			Float: CLASS_WEIGHT_0 -> Weight for class 0
			Float: CLASS_WEIGHT_1 -> Weight for class 1

		Returns: lambda function defined as: weighted_cross_entropy(y, y_pred, weights) + jaccard_loss(y, y_pred)
		"""
		weights = torch.Tensor([CLASS_WEIGHT_0, CLASS_WEIGHT_1])
		criterion = lambda y, y_pred: ce_loss(y, y_pred, weights.to(device)) + jaccard_loss(y, y_pred)
		return criterion

	def init_augmentation(self, CONFIGURATION):
		"""
		Initialization of augmentation function
			Int: CROP_SIZE_HEIGHT -> Height of Image crop
			Int: CROP_SIZE_WIDTH -> Width of Image crop
			Float: VERTICAL_FLIP_PROBA -> Probability of vertical flipping the image
			Float: HORIZONTAL_FLIP_PROBA -> Probability of horizontal flipping the image
			Float: GAUSS_NOISE_PROBA -> Probability of preforming gauss noise to image
			Float: GAUSS_NOISE_MAX -> Maximum value of gauss noise

		Returns: Augmentation function (albumentations.Compose)
		"""
		augmentation = albumentations.Compose([
			albumentations.RandomCrop(CONFIGURATION.CROP_SIZE_HEIGHT, CONFIGURATION.CROP_SIZE_WIDTH, p=1.0),
			albumentations.HorizontalFlip(p=CONFIGURATION.HORIZONTAL_FLIP_PROBA),
			albumentations.VerticalFlip(p=CONFIGURATION.VERTICAL_FLIP_PROBA),
			albumentations.GaussNoise(var_limit=CONFIGURATION.GAUSS_NOISE_MAX, p=CONFIGURATION.GAUSS_NOISE_PROBA),
			albumentations.Cutout(num_holes=CONFIGURATION.NUM_HOLES, max_h_size=CONFIGURATION.HOLE_SIZE, max_w_size=CONFIGURATION.HOLE_SIZE, p=CONFIGURATION.CUTOUT_PROBA),
			albumentations.Blur(blur_limit=6, p=0.6),
			albumentations.OneOf([
				albumentations.RandomBrightness(p=0.5),
				albumentations.RandomBrightnessContrast(p=0.5),				
				], p=0.4),
			HorizontalShift(shift_width=(3, 6), shift_height=(2, 4), shift_p=0.1,  always_apply=False, p=0.1),
			InvertImg(p=0.5)	
			])

		return augmentation

	def get_data(self, path, bordername='borders.npy', seismicname='seismic.npy'):
		"""
		Reading data from files.
			Str: path -> Path to folder with arrays

		Returns: seismic, borders data
		"""	
		try:
			x_data = np.load(path + seismicname)
		except:
			raise FileNotFoundError(f'[ERR] File {path + seismicname} with seismic data does not exist')

		try:
			y_data = np.load(path + bordername)
		except:
			print(f'[WARN] File {path + bordername} with labels data does not exist')
			y_data = None

		return x_data, y_data

	def get_dataloader(self, seismic, borders, aug, batch_size, shuffle, dtype='Train'):
		"""
		Creates dataloader instance.
			np.array: seismic -> Numpy array with seismic data
			np.array: borders -> Numpy array with masks
			albumentations.Compose: aug -> Augmentations
			Int: batch_size -> Batch size for dataloader
			Bool: shuffle -> Shuffle data

		Returns: torch.Dataloader
		"""

		dataset = SeismicDataset(seismic, borders, aug, dtype=dtype)

		dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)

		return dataloader

	def init_device(self, DEVICE):
		"""
		Initialization of torch.Device.
			Str: Device -> Device type (cpu / cuda:N)

		Returns: torch.Device
		"""

		device = torch.device(DEVICE)
		return device

	def init_metric(self):
		"""
		Initialization of IoU Metric
			(Implementation from https://www.kaggle.com/iezepov/fast-iou-scoring-metric-in-pytorch-and-numpy)

		Returns: lambda function that computes IoU Metric.
		"""

		return lambda y, y_pred: iou_pytorch(y_pred.cpu(), y.cpu())

	def train(self):
		CONFIGURATION = self.CONFIGURATION
		seismic, borders = self.get_data(CONFIGURATION.TRAIN_PATH)
		self.dataloader = self.get_dataloader(seismic, borders, self.aug, CONFIGURATION.BATCH_SIZE_TRAIN, CONFIGURATION.SHUFFLE_TRAIN)
		self.train_loop(self.model, self.optimizer, self.criterion, self.metric, self.dataloader, self.device, LOGGER, CONFIGURATION.NUM_EPOCHS, CONFIGURATION.CHECKPOINT_EVERY_N_EPOCHS, CONFIGURATION.MODEL_SAVE_PATH)
		LOGGER.write_to_file()

	def train_loop(self, model, optimizer, criterion, metric, dataloader, device, logger, NUM_EPOCHS, CHECKPOINT_EP, CHECKPOINT_DIR):
		"""
		Training loop for model.
			model -> Torch model to train
			optimizer -> Model optimizer
			criterion -> loss function
			torch.DataLoader: dataloader -> Data loader instance
			torch.device: device -> Device (GPU / CPU)
			Int: NUM_EPOCHS -> Number of epoch to train

		Returns: None
		"""

		model.to(device)
		for epoch in range(NUM_EPOCHS):
			for X, Y in iter(dataloader):

				#Null gradients
				optimizer.zero_grad()

				#Data to device
				X = X.to(device)
				Y = Y.to(device)

				#Make predictions
				Y_pred = model(X.unsqueeze(1))


				#Compute loss
				loss = criterion(Y.long(), Y_pred)

				#Compute metrics
				metrics = metric(Y.long(), Y_pred.argmax(1))

				#Backprop loss
				loss.backward()

				#Gradient step
				optimizer.step()

				#Write metrics and loss to Logger
				logger.write_metric(float(metrics.cpu()))
				logger.write_loss(float(loss.detach().cpu()))

			if (epoch + 1) % CHECKPOINT_EP == 0:
				torch.save(model.state_dict(), CHECKPOINT_DIR + f'{epoch}ep.torch')
		return None

	def inference(self, path, name):
		seismic, borders = self.get_data(path)
		self.dataloader = self.get_dataloader(seismic, borders, False, 1, False, dtype='Valid')
		self.inference(self.model, self.dataloader, self.device, SAVE=True, SAVE_PREFIX=name)

	def inference(self, model, dataloader, device, SAVE=True, SAVE_PREFIX='Valid'):
		"""
		Validating loop for model.
			model -> Torch model to validate / test
			criterion -> loss function
			torch.DataLoader: dataloader -> Data loader instance
			torch.device: device -> Device (GPU / CPU)
			Int: NUM_EPOCHS -> Number of epoch to validate

		Returns: None
		"""
		model.to(device)
		for i, (X, Y) in enumerate(iter(dataloader)):

			#Data to device
			X = X.to(device)
			Y = Y.to(device)

			#Make predictions
			Y_pred = model(X.unsqueeze(1))

			if i == 0:
				output = Y_pred.argmax(1).cpu().numpy().astype(np.uint8)
			else:
				output = np.concatenate([output, Y_pred.argmax(1).cpu().numpy().astype(np.uint8)], axis=0)

		if SAVE:
			np.save(f'output/predicted/{SAVE_PREFIX}.npy', output)

		return None