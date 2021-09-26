import torch
from torch import nn
import math
import sys
sys.path.append('../global_module/')
from torch.nn import ReLU, Parameter, Softmax
from activation import mish
torch_ver = torch.__version__[:3]

class DENSE(nn.Module):
    def __init__(self, band, classes):
        super(DENSE, self).__init__()

        # spectral branch
        self.name = 'SSAC'
        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
                                    nn.BatchNorm3d(24,  eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
                                    nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),

            mish()
        )
        self.conv13 = nn.Conv3d(in_channels=36, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
                                    nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),

            mish()
        )
        self.conv14 = nn.Conv3d(in_channels=48, out_channels=12, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
                                    nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            mish()
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=60, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))


        self.batch_norm_spectral = nn.Sequential(
                                    nn.BatchNorm3d(60,  eps=0.001, momentum=0.1, affine=True),
            mish(),
                                    nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
                                nn.Linear(60, classes)
        )

    def forward_once(self, X):
        # spectral
        x11 = self.conv11(X)
        #print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        #print('x12', x12.shape)
        x13 = torch.cat((x11, x12), dim=1)
        #print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        #print('x13', x13.shape)
        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)
        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)
        x16 = self.batch_norm14(x15)
        output = self.conv15(x16)

        return output

    def forward(self, X1, X2):
        output1 = self.forward_once(X1)
        output2 = self.forward_once(X2)
        return output1, output2

class CAM(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim, classes):
        super(CAM, self).__init__()
        self.chanel_in = in_dim
        self.relu = ReLU(inplace=True)
        self.name = 'SSAC_CAM'
        self.gamma = Parameter(torch.zeros(1))
        self.softmax  = Softmax(dim=-1)
        self.batch_norm_spectral = nn.Sequential(
            nn.BatchNorm3d(60, eps=0.001, momentum=0.1, affine=True),
            # gelu_new(),
            # swish(),
            mish(),
            nn.Dropout(p=0.5)
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            # nn.Dropout(p=0.5),
            nn.Linear(60, classes)
            # nn.Softmax()
        )
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy

        attention = self.softmax(energy_new)

        proj_value = x.view(m_batchsize, C, -1)
        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        out = out.cpu()
        x = x.cpu()
        out = self.gamma*out + x  #C*H*W
        x1 = torch.mul(out, x)
        x1 = self.batch_norm_spectral(x1)
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        output = self.full_connection(x1)
        output = output.cuda()
        return output, attention