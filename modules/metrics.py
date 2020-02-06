import torch
def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor, SMOOTH=0.001, mean=True):

    intersection = (outputs & labels).float().sum((1, 2))  
    union = (outputs | labels).float().sum((1, 2))         
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)

    if mean:
        return iou.mean()
    else:
        return iou