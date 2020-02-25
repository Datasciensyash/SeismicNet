import streamlit as st 
import numpy as np
import os
from trainrunner import TrainRunner 

import pandas as pd
from modules.metrics import iou_numpy
from modules.config import Config
import albumentations
import matplotlib.pyplot as plt 
@st.cache
def iou_score(mask, predicted, tr=0.5):
    return iou_numpy(predicted, mask, mean=False, THRESHOLD=tr)


@st.cache
def read_logs(cfn):
	cf = Config(cfn)
	metricsfile = cf.LOGDIR + cf.MODEL_NAME + '-metrics-train.txt'
	lossfile = cf.LOGDIR + cf.MODEL_NAME + '-loss-train.txt'

	with open(metricsfile) as f:
		metrics = list(map(float, f.read().split(' ')[:len(f.read()) - 1]))

	with open(lossfile) as f:
		loss = list(map(float, f.read().split(' ')[:len(f.read()) - 1]))
	return loss, metrics

def visualize_metrics(metrics, loss, plot=False, title='0'):
	if not plot:
		st.line_chart(pd.DataFrame(list(zip(metrics, loss)), columns=['IoU', 'Loss']))
	if plot:
		plt.figure(figsize=(10, 4))
		plt.title(title)
		plt.plot([i for i in range(len(loss))], loss, c='r', alpha=0.8, label='loss')
		plt.plot([i for i in range(len(metrics))], metrics, c='g', alpha=0.8, label='IoU')
		plt.axhline(1, c='gray', label=f'Maximum possible val. of IoU', alpha=0.15)
		plt.xlabel('Batch num')
		plt.ylabel('Value')
		plt.legend()
		plt.savefig('output/' + title + '-train.png')
		st.pyplot()
		plt.cla()

def visualize_predictions(metrics, plot=False, title='0'):
	if not plot:
		st.line_chart(pd.DataFrame(metrics, columns=['IoU']))
	if plot:
		plt.figure(figsize=(10, 4))
		plt.title('Val. of ' + title)
		plt.plot([i for i in range(len(metrics))], metrics, c='g', alpha=0.8, label='IoU')
		plt.axhline(np.max(metrics), c='g', label=f'Maximum val. of IoU {np.round(metrics.max(), 3)} ± {np.round(metrics.std(), 4)}', alpha=0.15)
		plt.axhline(np.min(metrics), c='r', label=f'Minimum val. of IoU {np.round(metrics.min(), 3)}', alpha=0.15)
		plt.axhline(np.mean(metrics), c='b', label=f'Mean val. of IoU {np.round(metrics.mean(), 3)}', alpha=0.2)			
		plt.xlabel('Slice num')
		plt.ylabel('Value')
		plt.legend()
		plt.savefig('output/' + title + '-pred.png')
		st.pyplot()
		plt.cla()

@st.cache
def predict_and_eval_score(configfile, data_path='data/valid/', seismic_name='seismic.npy', val_mask='horizons_1.npy'):
	TrainRunner(configfile).predict(data_path, seismic_name, suffix='valid')
	config = Config(configfile)
	mask = np.load(f'data/valid/{val_mask}').astype(np.uint8)
	predicted_mask = np.load('output/predictions/' + config.MODEL_NAME + '-valid.npy')
	return iou_score(mask, predicted_mask, 0.5)


exp = st.sidebar.slider('Num of experiments', 1, 20)
train = st.sidebar.checkbox('Train', value=False)

experiments = []
for i in range(exp):
	et = st.sidebar.text_input(f'Experiment {i} title', value='Horizon thickness 1')
	config = st.sidebar.text_input(f'[Train] Configfile {i}', value='configs/')
	pconfig = st.sidebar.text_input(f'[Predict] Configfile {i}', value='configs/')
	experiments.append([et, config, pconfig])

run = st.sidebar.button('Run experiments')

if run:
	for experiment in experiments:
		if train:
			TrainRunner(experiment[1]).train('data/train/')
		l, m = read_logs(experiment[1])
		st.header(f'Experiment {experiment[0]}')
		visualize_metrics(m, l, True, title=experiment[0])
		scores = predict_and_eval_score(experiment[2], val_mask=Config(experiment[2]).MASK_FILENAME)
		visualize_predictions(scores, True, title=experiment[0])
		st.info(f'Mean value: {scores.mean()} +/- {scores.std()}')
		st.info(f'Max value: {scores.max()}')
		st.info(f'Min value: {scores.min()}')		









