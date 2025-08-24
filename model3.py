import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange

class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=5,
                              padding='same')
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm1d(out_channels)
        
    def forward(self,x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.layer1 = ConvBNReLU(in_channels=1, out_channels=32)
        self.layer2 = ConvBNReLU(in_channels=32, out_channels=64)
        self.layer3 = ConvBNReLU(in_channels=64, out_channels=128)
        
        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 3)
        
        self.mp1 = nn.MaxPool1d(kernel_size=2)
        self.mp2 = nn.MaxPool1d(kernel_size=2)
        
        self.relu = nn.ReLU()
        
        self.softmax = nn.Softmax(dim=1)
        
        self.dropout1 = nn.Dropout(p=0.5)
        self.dropout2 = nn.Dropout(p=0.5)
        self.dropout3 = nn.Dropout(p=0.5)
        
        self.gap = nn.AdaptiveAvgPool1d(1)
        
        
    def forward(self, x):
        

        x = self.layer1(x)
        x = self.mp1(x)
        x = self.layer2(x)
        x = self.mp2(x)
        x = self.layer3(x)
        
        x = self.gap(x)
        x = rearrange(x, 'B C L -> B (C L)')
        
        x = self.dropout1(x)
        
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.dropout2(x)
        
        x = self.linear2(x)
        x = self.relu(x)
        
        x = self.dropout3(x)
        
        x = self.linear3(x)
        x = self.softmax(x)
        return x
    
    
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model = Model().to('cpu')
    summary(model, input_size=(1, 1, 128))
        