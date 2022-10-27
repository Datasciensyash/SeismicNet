import torch
import numpy as np

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, SMOOTH=0.001, mean=True):

    intersection = (outputs & labels).float().sum((1, 2))
    union = (outputs | labels).float().sum((1, 2))         

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean() if mean else iou

def iou_numpy(outputs: np.array, labels: np.array, SMOOTH=0.001, mean=True, THRESHOLD=0.5):


    #Compute mask by THRESHOLD:
    outputs = np.round(outputs - (THRESHOLD - 0.5)).astype(np.uint8)

    intersection = (outputs & labels).sum((1, 2))
    union = (outputs | labels).sum((1, 2))

    iou = (intersection + SMOOTH) / (union + SMOOTH)

    return iou.mean() if mean else iou