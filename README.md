# H-Net
Fully convolutional network for seismic horizon detection.

![Header](images/header.png)

---

## Problem statement
Seismic horizons are the borders between two rock layers with different physical properties. The task is to find these horizons all over the seismic cube and perform the binary segmentation: *does this pixel is horizon or not*.
![Problem statement](images/ps.png)
Currently, the task of finding seismic horizons is solved by classical computer vision methods, which require preliminary marking of __each new seismic cube__ and __often break on a complex underground relief__.

In this work, we introduce a new approach to this task using deep learning methods.

## Reproducibility
```
pip install -r requirements.txt
python download.py
python train.py --config config.txt
python inference.py --config config.txt --path data/valid/
streamlit run explore.py
```
