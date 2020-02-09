import streamlit as st 
import numpy as np
import os

import pandas as pd
from modules.metrics import iou_numpy

@st.cache
def iou_score(mask, predicted, tr=0.5):
	return iou_numpy(predicted, mask, mean=False, THRESHOLD=tr)


def visualize_metrics(metrics):
	st.line_chart(pd.DataFrame(metrics, columns=['IoU']))

def visualize_ttm(mean, std, index):
	st.line_chart(pd.DataFrame(list(zip(mean, std)), columns=['IoU Mean', 'IoU Std'], index=index))


@st.cache
def min_max_norm(array):
	return (array - array.min()) / (array.max() - array.min())

@st.cache
def threshold_mask(mask, tr):
	return np.round(mask - (tr - 0.5)).astype(np.uint8)

@st.cache
def load_array(ORIGINAL_DATA_SEISMIC, ORIGINAL_DATA_MASK, OUT_DATA_MASK_DIR, OUT_DATA_MASK):
	seismic = min_max_norm(np.load(ORIGINAL_DATA_SEISMIC))
	mask = (1 - np.load(ORIGINAL_DATA_MASK)).astype('uint8')
	predicted = (1 - np.load(OUT_DATA_MASK_DIR + OUT_DATA_MASK))
	return seismic, mask, predicted

@st.cache
def threshold_to_metrics(mask, predicted, minv, maxv, step):
	tr_list = np.linspace(minv, maxv, (maxv-minv) // step + 1)
	ttm_mean = []
	ttm_std = []
	for tr in tr_list:
		ttm_mean.append(iou_score(mask, predicted, tr).mean())
		ttm_std.append(iou_score(mask, predicted, tr).std())
	return ttm_mean, ttm_std, tr_list

@st.cache
def get_distribution(mask, minv, maxv, step):
	distribution = []
	index = []
	for i in range(0, int((maxv - minv) // step) + 1):
		distribution.append(np.sum((mask > i * step) & (mask < (i + 1) * step)))
		index.append(f'{round(i * step, 3)} - {round((i + 1) * step, 3)}')
	return distribution, index

def plot_distribution(distribution, index):
	chart_data = pd.DataFrame(distribution, columns=["Number of predictions"], index=index)
	st.bar_chart(chart_data)

@st.cache
def get_error_val(mask, predictions):
	mean = []
	std = []
	mean = np.mean(predictions[mask != predictions])
	std = np.std(predictions[mask != predictions])
	return mean, std




ORIGINAL_DATA_SEISMIC = st.sidebar.text_input('Path to original seismic data', value='data/valid/seismic.npy')
ORIGINAL_DATA_MASK = st.sidebar.text_input('Path to original mask data', value='data/valid/borders.npy')

OUT_DATA_MASK_DIR = st.sidebar.text_input('Path to output mask data directory', value='output/predictions/')
OUT_DATA_MASK = st.sidebar.selectbox('Select filename', os.listdir(OUT_DATA_MASK_DIR))


SHOW_GRAPH = st.sidebar.checkbox('Show statistics', value=False)
SHOW_TTM_GRAPH = st.sidebar.checkbox('Show ttm graph', value=False)
SHOW_IMAGES = st.sidebar.checkbox('Show images', value=False)

#Loading arrays
seismic, mask, predicted = load_array(ORIGINAL_DATA_SEISMIC, ORIGINAL_DATA_MASK, OUT_DATA_MASK_DIR, OUT_DATA_MASK)
#

if SHOW_IMAGES or SHOW_GRAPH:
	tr = st.sidebar.slider('Threshold', min_value=0., max_value=1.0, step=0.01, value=0.5)

if SHOW_IMAGES:
	idx = st.sidebar.slider('Id of prediction', min_value=0, max_value=len(seismic) - 1)

if SHOW_GRAPH:
	metrics = iou_score(mask, predicted, tr)
	st.success(f'IoU mean: {metrics.mean()} ± {metrics.std()}')
	visualize_metrics(metrics)
	st.info(f'Std. of predictions: {predicted.std()}')
	step = st.slider('Step value', value=0.1)
	dis, index = get_distribution(predicted, 0, 1, step)
	plot_distribution(dis, index)
	if st.checkbox('Show error val?'):
		predictions = threshold_mask(predicted, tr)
		mean, std = get_error_val(mask, predictions)
		st.info(f'Err. mean val: {mean} ± {std}')

if SHOW_TTM_GRAPH:
    ttm_mean, ttm_std, index = threshold_to_metrics(mask, predicted, 0, 1, 0.01)
    st.success(f'Max. IoU: {round(np.max(ttm_mean), 4)} with THRESHOLD = {round(index[np.argmax(ttm_mean)], 4)}')
    visualize_ttm(ttm_mean, ttm_std, index)	


if SHOW_IMAGES:
	metrics = iou_score(mask, predicted, tr)
	mask = min_max_norm(mask)
	predicted = min_max_norm(threshold_mask(predicted, tr))
	st.image([seismic[idx].T, mask[idx].T, predicted[idx].T], caption=['Seismic data', 'Ground truth', f'Predicted with IoU {round(metrics[idx], 3)}'])

