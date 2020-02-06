import streamlit as st 
import numpy as np
import os

import pandas as pd
from modules.metrics import iou_numpy

@st.cache
def iou_score(mask, predicted):
	return iou_numpy(mask, predicted, mean=False)

@st.cache
def visualize_metrics(metrics):
	st.line_chart(pd.DataFrame(metrics, columns=['IoU']))

@st.cache
def min_max_norm(array):
	return (array - array.min()) / (array.max() - array.min())

ORIGINAL_DATA_SEISMIC = st.text_input('Path to original seismic data', value='data/valid/seismic.npy')
ORIGINAL_DATA_MASK = st.text_input('Path to original mask data', value='data/valid/borders.npy')

OUT_DATA_MASK_DIR = st.text_input('Path to output mask data directory', value='output/predictions/')
OUT_DATA_MASK = st.selectbox('Select filename', os.listdir(OUT_DATA_MASK_DIR))

#Loading arrays
seismic = min_max_norm(np.load(ORIGINAL_DATA_SEISMIC))
mask = np.load(ORIGINAL_DATA_MASK)
predicted = np.load(OUT_DATA_MASK_DIR + OUT_DATA_MASK)
#

metrics = iou_score(mask, predicted)
visualize_metrics(metrics)

mask = min_max_norm(mask)
predicted = min_max_norm(predicted)

idx = st.slider('Id of prediction', min_value=0, max_value=len(seismic) - 1)
st.image(seismic[idx].T)
st.image([mask[idx].T, predicted[idx].T], caption=['Ground truth', f'Predicted with IoU {round(metrics[idx], 3)}'])

