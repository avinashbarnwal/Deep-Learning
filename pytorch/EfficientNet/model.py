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

class EffNet(nn.Module):
    def __init__(self,model='b2'):
        super(EffNet,self).__init__()
        
        model_name = 'efficientnet' + model
        self.feature = EfficientNet.from_pretrained("efficientnet-b2")
        self.drop = nn.Dropout(0.3)
        self.l0 = nn.Linear(1408,1)
        
        
    def forward(self,img):
        batch_size = img.shape[0]
        
        x = self.feature.extract_features(img)
        #print(x.shape)
        
        x = nn.functional.adaptive_avg_pool2d(x,1).reshape(batch_size,-1)
        #print(x.shape)
        
        #x = self.drop(x)
        #print(x.shape)
        out = self.l0(x)
        #print(out.shape)
        
        return out