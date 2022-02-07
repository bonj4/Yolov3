# -*- coding: utf-8 -*-
"""
Created on Sat Jan 22 16:08:18 2022

@author: faruk
"""
import torch.nn as nn
import torch


def conv_batch(in_num, out_num, kernel_size=3, padding=1, stride=1,bias=False):
    if bias==False:
       return nn.Sequential(
        nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
        nn.BatchNorm2d(out_num),
        nn.LeakyReLU())
    else: 
       return nn.Sequential(nn.Conv2d(in_num, out_num, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),)

# Residual block
class DarkResidualBlock(nn.Module):
    def __init__(self, in_channels):
        super(DarkResidualBlock, self).__init__()

        reduced_channels = int(in_channels/2)

        self.layer1 = conv_batch(in_channels, reduced_channels, kernel_size=1, padding=0)
        self.layer2 = conv_batch(reduced_channels, in_channels)

    def forward(self, x):
        residual = x

        out = self.layer1(x)
        out = self.layer2(out)
        out += residual
        return out


class predict(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(predict, self).__init__()
        self.layer1 = conv_batch(in_channels, in_channels*2)
        self.layer2 = conv_batch(in_channels*2,( num_classes+5)*3, kernel_size=1,padding=0, bias=True)
        self.num_classes = num_classes
    def forward(self, resb):
        out = self.layer1(resb)
        out = self.layer2(out)
        return out.reshape(out.shape[0], 3, self.num_classes + 5, out.shape[2], out.shape[3]).permute(0, 1, 3, 4, 2)
    
            

class Darknet53(nn.Module):
    def __init__(self, block, num_classes):
        
        super(Darknet53, self).__init__()

        self.num_classes = num_classes

        self.conv1 = conv_batch(3, 32)
        self.conv2 = conv_batch(32, 64, stride=2)
        self.residual_block1 = self.make_layer(block, in_channels=64, num_blocks=1)
        self.conv3 = conv_batch(64, 128, stride=2)
        self.residual_block2 = self.make_layer(block, in_channels=128, num_blocks=2)
        self.conv4 = conv_batch(128, 256, stride=2)
        self.residual_block3 = self.make_layer(block, in_channels=256, num_blocks=8)
        self.conv5 = conv_batch(256, 512, stride=2)
        self.residual_block4 = self.make_layer(block, in_channels=512, num_blocks=8)
        self.conv6 = conv_batch(512, 1024, stride=2)
        self.residual_block5 = self.make_layer(block, in_channels=1024, num_blocks=4)
        self.load_pretrained_layers()



    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.residual_block1(out)
        out = self.conv3(out)
        out = self.residual_block2(out)
        out = self.conv4(out)
        out = self.residual_block3(out)
        res8_1=out
        out = self.conv5(out)
        out = self.residual_block4(out)
        res8_2=out
        out = self.conv6(out)
        out = self.residual_block5(out)
        res4_1=out
        return [res8_1, res8_2, res4_1]
    
    
    def make_layer(self,block, in_channels, num_blocks):
        layer=[]
        for i in range(num_blocks):
            layer.append(block(in_channels))
        return nn.Sequential(*layer)
    def load_pretrained_layers(self):
        # Current state of base
        state_dict = self.state_dict()
        param_names = list(state_dict.keys())

        # Pretrained VGG base
        pretrained_state_dict = torch.load("darknet53.pth.tar")['state_dict']
        pretrained_param_names = list(pretrained_state_dict.keys())

        # Transfer conv. parameters from pretrained model to current model
        for i, param in enumerate(param_names):  # excluding conv6 and conv7 parameters
            state_dict[param] = pretrained_state_dict[pretrained_param_names[i]]


        self.load_state_dict(state_dict)

        print("\nLoaded base model.\n")



class yolov3(nn.Module):
    def __init__(self,num_classes):
        super(yolov3, self).__init__()
        self.base=Darknet53(DarkResidualBlock,num_classes)
        for param in self.base.parameters():
              	param.requires_grad = True

        self.convset4=self.convset(1024)
        self.pred4=predict(512,num_classes)
        self.upsample1=self.upsample(512)

        self.convset8_2=self.convset(768,512)
        self.pred8_2=predict(256,num_classes)
        self.upsample2=self.upsample(256)
        self.convset8_1=self.convset(384,256)
        self.pred8_1=predict(128,num_classes)

    def forward(self,image):
        outputs=[]
        out=self.base(image)
        out[2]=self.convset4(out[2])
        outputs.append(self.pred4(out[2]))

        out[1]=torch.cat([self.upsample1(out[2]),out[1]], dim=1)
        out[1]=self.convset8_2(out[1])
        outputs.append(self.pred8_2(out[1])) 

        out[0]=torch.cat([self.upsample2(out[1]),out[0]], dim=1)    
        out[0]=self.convset8_1(out[0])
        outputs.append(self.pred8_1(out[0]))
    
        return outputs
    def upsample(self,in_channels):
        # print(in_channels)
        layer=[]
        layer.append(conv_batch(in_channels,in_channels//2, kernel_size=1, padding=0))
        layer.append(nn.Upsample(scale_factor=2))
        return nn.Sequential(*layer)
        
    def convset(self,in_channels,y=None):
        layer=[]
        if y== None:
            y=in_channels
        layer+=conv_batch(in_channels,y//2,kernel_size=1,padding=0)
        layer+=conv_batch(y//2, y)
        layer+=conv_batch(y, y//2,kernel_size=1,padding=0)
        layer+=conv_batch(y//2, y)
        layer+=conv_batch(y, y//2,kernel_size=1,padding=0)
        # print(nn.Sequential(*layer))
        return nn.Sequential(*layer)

if __name__ == "__main__":
    num_classes = 20
    IMAGE_SIZE = 416
    model = yolov3(num_classes=num_classes)
    x = torch.randn((2, 3, IMAGE_SIZE, IMAGE_SIZE))
    out = model(x)
    assert model(x)[0].shape == (2, 3, IMAGE_SIZE//32, IMAGE_SIZE//32, num_classes + 5)
    assert model(x)[1].shape == (2, 3, IMAGE_SIZE//16, IMAGE_SIZE//16, num_classes + 5)
    assert model(x)[2].shape == (2, 3, IMAGE_SIZE//8, IMAGE_SIZE//8, num_classes + 5)
    print("Success!")
    #torch.split(x,x.size()[1]//2,dim=1)[0].size()