import streamlit as st 
import numpy as np
import os
from trainrunner import TrainRunner 

import pandas as pd
from modules.metrics import iou_numpy
from modules.config import Config
import albumentations

@st.cache
def iou_score(mask, predicted, tr=0.5):
    return iou_numpy(predicted, mask, mean=False, THRESHOLD=tr)

def change_config_file(variable, val, configfile='config.txt', out_config='configs/config.txt'):
	cf = Config(configfile)
	for i in range(len(variable)):
		cf.config[variable[i]] = val[i]
	cf.write_config_file(out_config)
	return out_config

exp = st.sidebar.slider('Num of experiments', 1, 20)
mainconfig = st.sidebar.text_input('Configfile', value='config.txt')

exps = []
for experiment in range(exp):
	name = st.text_input(f'Exp. {experiment} name', value='BS-')
	train = st.selectbox(f'Exp. {experiment} train_mask', os.listdir('data/train/'))
	test = st.selectbox(f'Exp. {experiment} valid seismic', os.listdir('data/valid/'))
	exps.append([name, train, test])
	st.write('-------------------------------------------------')

if st.sidebar.button('RUN'):
	for ex in exps:
		cfn = change_config_file(['MODEL_NAME', 'LOAD_MODEL', 'MASK_FILENAME', 'LOGNAME'], [f'U-NET-{ex[0]}', False, ex[1], ex[0]])
		TrainRunner(cfn).train('data/train/')
		cfn = change_config_file(['LOAD_MODEL'], [True], configfile=cfn)
		TrainRunner(cfn).predict('data/valid/', ex[2], ex[0])
	st.success('Got it')











