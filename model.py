import torch
import torch.nn as nn
import torch.nn.functional as F


def deconv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom deconvolutional layer for simplicity."""
    layers = []
    layers.append(nn.ConvTranspose2d(c_in, c_out, k_size, stride, pad, bias=True))#bias=False
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

def conv(c_in, c_out, k_size, stride=2, pad=1, bn=True):
    """Custom convolutional layer for simplicity."""
    layers = []
    layers.append(nn.Conv2d(c_in, c_out, k_size, stride, pad, bias=True))#bias=False
    if bn:
        layers.append(nn.BatchNorm2d(c_out))
    return nn.Sequential(*layers)

class G(nn.Module):
    """Generator for transfering from mnist to svhn"""
    def __init__(self, conv_dim=64):
        super(G, self).__init__()
        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim*2, 5, 2, 2)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 5, 2, 2)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 1, 0)
        self.fc = conv(conv_dim*8, 10, 1, 1, 0, bn=False)
        
        self.deconv1 = deconv(conv_dim*8 + 2, conv_dim*4, 4, 4, 0)
        self.deconv2 = deconv(conv_dim*4, conv_dim*2, 4)
        self.deconv3 = deconv(conv_dim*2, conv_dim, 4)
        self.deconv4 = deconv(conv_dim  , 3, 4, bn=False)
        
    def forward(self, x, code):
        outc = F.leaky_relu(self.conv1(x), 0.05)      
        outc = F.leaky_relu(self.conv2(outc), 0.05)    
        
        outc = F.leaky_relu(self.conv3(outc), 0.05)    
        outc = F.leaky_relu(self.conv4(outc), 0.05)    
        outcc = self.fc(outc).squeeze()
        out = torch.cat((outc, code), 1)
        
        out = F.leaky_relu(self.deconv1(out), 0.05)  
        out = F.leaky_relu(self.deconv2(out), 0.05)
        out = F.leaky_relu(self.deconv3(out), 0.05)
        out = F.tanh(self.deconv4(out))         
        return out, outc, outcc
    

    
class D1(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64):
        super(D1, self).__init__()
        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim*2, 5, 2, 2)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 5, 2, 2)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 1, 0)
        self.fc1 = conv(conv_dim*8, 11, 1, 1, 0, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)   
        out = F.leaky_relu(self.conv2(out), 0.05) 
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = self.fc1(out).squeeze()
        
        return out
class D2(nn.Module):
    """Discriminator for mnist."""
    def __init__(self, conv_dim=64):
        super(D2, self).__init__()
        self.conv1 = conv(3, conv_dim, 5, 2, 2)
        self.conv2 = conv(conv_dim, conv_dim*2, 5, 2, 2)
        self.conv3 = conv(conv_dim*2, conv_dim*4, 5, 2, 2)
        self.conv4 = conv(conv_dim*4, conv_dim*8, 4, 1, 0)
        self.fc1 = conv(conv_dim*8, 11, 1, 1, 0, bn=False)
        
    def forward(self, x):
        out = F.leaky_relu(self.conv1(x), 0.05)   
        out = F.leaky_relu(self.conv2(out), 0.05) 
        out = F.leaky_relu(self.conv3(out), 0.05)
        out = F.leaky_relu(self.conv4(out), 0.05)
        out = self.fc1(out).squeeze()
        
        return out
