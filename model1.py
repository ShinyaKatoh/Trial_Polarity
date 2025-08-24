import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
from einops import rearrange
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        
        in_ch = 128
        
        self.layer1 = nn.Linear(in_ch, in_ch)    
        self.layer2 = nn.Linear(in_ch, in_ch)  
        self.layer3 = nn.Linear(in_ch, in_ch) 
        
        self.linear1 = nn.Linear(in_ch, in_ch//2) 
        self.linear2 = nn.Linear(in_ch//2, in_ch//4) 
        self.linear3 = nn.Linear(in_ch//4, 3) 
        
        self.relu = nn.ReLU()
        
        self.softmax = nn.Softmax(dim=1)
        
    def forward(self, x):
        
        x = rearrange(x, 'b c l -> (b c) l')
        
        x = self.layer1(x)
        x = self.relu(x)
        
        x = self.layer2(x)
        x = self.relu(x)
        
        x = self.layer3(x)
        x = self.relu(x)
    
        x = self.linear1(x)
        x = self.relu(x)
        
        x = self.linear2(x)
        x = self.relu(x)
       
        x = self.linear3(x)
        x = self.softmax(x)
        return x
    
    
if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    
    model = Model().to('cpu')
    summary(model, input_size=(1, 1, 128))
        