# H-Net
Fully convolutional network for seismic horizon detection.

![Header](images/header.png)

---
## Problem statement
Seismic horizons are the borders between two rock layers with different physical properties. The task is to find these horizons all over the seismic cube and perform the binary segmentation: *does this pixel is horizon or not*.
![Problem statement](images/ps.png)
Currently, the task of finding seismic horizons is solved by classical computer vision methods, which require preliminary marking of __each new seismic cube__ and __often break on a complex underground relief__.

In this work, we introduce a new approach to this task using deep learning methods.

---
## Dataset
For training and validating we will use [Netherlands F3 dataset](https://github.com/olivesgatech/facies_classification_benchmark) containing the seismic cube and marked seismic facies. Seismic facies is the rock bodies between two seismic horizons, so we can easily perform transformation between facies and horizons. 

[IMAGE WITH TRANSFORMATION]

Since the seismic cube is 3D data we can get two types of vertical sections along two axes. In seismic exploration, they are called inlines and crosslines. This is a very useful property that allows you to get more training examples.

## Reproducibility
```
pip install -r requirements.txt
python download.py
python train.py --config config.txt
python inference.py --config config.txt --path data/valid/
streamlit run explore.py
```
