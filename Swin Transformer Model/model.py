import torch.nn as nn
from torchvision.models import swin_t

class SwinTransformerChestXray(nn.Module):
    def __init__(self, num_classes=2):
        super(SwinTransformerChestXray, self).__init__()
        self.swin = swin_t(weights='IMAGENET1K_V1')
        
        # Freeze the feature extractor
        for param in self.swin.parameters():
            param.requires_grad = False
        
        # Replace the last fully connected layer
        num_features = self.swin.head.in_features
        self.swin.head = nn.Linear(num_features, num_classes)
    
    def forward(self, x):
        return self.swin(x)

def create_model(num_classes=2):
    return SwinTransformerChestXray(num_classes)