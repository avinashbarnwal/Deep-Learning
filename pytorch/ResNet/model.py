import torch
import torch.nn as nn
import torchvision.models as models
import torchvision
import PIL
import cv2
from   torch.utils.data import DataLoader,TensorDataset,Dataset
import matplotlib.pyplot as plt
from   sklearn import model_selection
from   sklearn.metrics import roc_auc_score
from   efficientnet_pytorch import EfficientNet


class Resnext50_32x4d(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = models.resnext50_32x4d(pretrained = True)
        self.l1 = nn.Linear(1000,1)
        
    def forward(self,image,view=True):
        #if view==True : print("Image shape {}".format(image.shape))
            
        img = self.model(image)
        
        #if view == True : print('Resnext50_32x4d output shape {}'.format(img.shape))
        
        out = self.l1(img)
        #print("Output Shape {}".format(out.shape))
        
        return out
