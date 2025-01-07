from torch import nn
import torch

class ATT3DLayer(nn.Module):
    def __init__(self, channel, reduction=18):
        super(ATT3DLayer, self).__init__()
        self.channel = channel
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(reduction, channel * 3, bias=False),
            # nn.Sigmoid()
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        b, c, _, _, _ = x.size()
        y = self.avg_pool(x) 
        y = y.view(b, c) 
        y = self.fc(y)
        y = y.view(b, c*3, 1, 1, 1)
        y1, y2, y3 = torch.split(y, [self.channel, self.channel, self.channel], dim=1)
        return y1,y2,y3