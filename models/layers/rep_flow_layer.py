"""
From
AJ Piergiovanni and Michael S. Ryoo
"Representation Flow for Action Recognition"
in CVPR 2019


More details: https://github.com/piergiaj/representation-flow-cvpr19

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlowLayer(nn.Module):

    def __init__(self, channels=1, params=[1,1,1,1,1], n_iter=10):
        super(FlowLayer, self).__init__()
        self.n_iter = n_iter
        if params[0]:
            self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,1,1,1))
            self.img_grad2 = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).transpose(3,2).repeat(channels,1,1,1))
        else:
            self.img_grad = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.img_grad2 = nn.Parameter(torch.FloatTensor([[[[-0.5,0,0.5]]]]).transpose(3,2).repeat(channels,1,1,1), requires_grad=False)            

            
        if params[1]:
            self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1))
            self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1))
            self.div = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1))
            self.div2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1))            
        else:
            self.f_grad = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.f_grad2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.div = nn.Parameter(torch.FloatTensor([[[[-1,1]]]]).repeat(channels,1,1,1), requires_grad=False)
            self.div2 = nn.Parameter(torch.FloatTensor([[[[-1],[1]]]]).repeat(channels,1,1,1), requires_grad=False)

            
        self.channels = channels
        
        self.t = 0.3
        self.l = 0.15
        self.a = 0.25        

        if params[2]:
            self.t = nn.Parameter(torch.FloatTensor([self.t]))
        if params[3]:
            self.l = nn.Parameter(torch.FloatTensor([self.l]))
        if params[4]:
            self.a = nn.Parameter(torch.FloatTensor([self.a]))


    def norm_img(self, x):
        mx = torch.max(x)
        mn = torch.min(x)
        x = 255*(x-mn)/(mn-mx)
        return x
            
    def forward_grad(self, x):
        grad_x = F.conv2d(F.pad(x, (0,1,0,0)), self.f_grad, groups=self.channels)
        grad_x[:,:,:,-1] = 0
        
        grad_y = F.conv2d(F.pad(x, (0,0,0,1)), self.f_grad2, groups=self.channels)
        grad_y[:,:,-1,:] = 0
        return grad_x, grad_y


    def divergence(self, x, y):
        tx = F.pad(x[:,:,:,:-1], (1,0,0,0))
        ty = F.pad(y[:,:,:-1,:], (0,0,1,0))
        
        grad_x = F.conv2d(F.pad(tx, (0,1,0,0)), self.div, groups=self.channels)
        grad_y = F.conv2d(F.pad(ty, (0,0,0,1)), self.div2, groups=self.channels)
        return grad_x + grad_y
        
        
    def forward(self, x, y):
        u1 = torch.zeros_like(x)
        u2 = torch.zeros_like(x)
        l_t = self.l * self.t
        taut = self.a/self.t

        grad2_x = F.conv2d(F.pad(y,(1,1,0,0)), self.img_grad, groups=self.channels, padding=0, stride=1)
        grad2_x[:,:,:,0] = 0.5 * (x[:,:,:,1] - x[:,:,:,0])
        grad2_x[:,:,:,-1] = 0.5 * (x[:,:,:,-1] - x[:,:,:,-2])

        
        grad2_y = F.conv2d(F.pad(y, (0,0,1,1)), self.img_grad2, groups=self.channels, padding=0, stride=1)
        grad2_y[:,:,0,:] = 0.5 * (x[:,:,1,:] - x[:,:,0,:])
        grad2_y[:,:,-1,:] = 0.5 * (x[:,:,-1,:] - x[:,:,-2,:])
        
        p11 = torch.zeros_like(x.data)
        p12 = torch.zeros_like(x.data)
        p21 = torch.zeros_like(x.data)
        p22 = torch.zeros_like(x.data)

        gsqx = grad2_x**2
        gsqy = grad2_y**2
        grad = gsqx + gsqy + 1e-12

        rho_c = y - grad2_x * u1 - grad2_y * u2 - x
        
        for i in range(self.n_iter):
            rho = rho_c + grad2_x * u1 + grad2_y * u2 + 1e-12

            v1 = torch.zeros_like(x.data)
            v2 = torch.zeros_like(x.data)
            mask1 = (rho < -l_t*grad).detach()
            v1[mask1] = (l_t * grad2_x)[mask1]
            v2[mask1] = (l_t * grad2_y)[mask1]
            
            mask2 = (rho > l_t*grad).detach()
            v1[mask2] = (-l_t * grad2_x)[mask2]
            v2[mask2] = (-l_t * grad2_y)[mask2]

            mask3 = ((mask1^1) & (mask2^1) & (grad > 1e-12)).detach()
            v1[mask3] = ((-rho/grad) * grad2_x)[mask3]
            v2[mask3] = ((-rho/grad) * grad2_y)[mask3]
            del rho
            del mask1
            del mask2
            del mask3

            v1 += u1
            v2 += u2

            u1 = v1 + self.t * self.divergence(p11, p12)
            u2 = v2 + self.t * self.divergence(p21, p22)
            del v1
            del v2
            u1 = u1
            u2 = u2

            u1x, u1y = self.forward_grad(u1)
            u2x, u2y = self.forward_grad(u2)
            
            p11 = (p11 + taut * u1x) / (1. + taut * torch.sqrt(u1x**2 + u1y**2 + 1e-12))
            p12 = (p12 + taut * u1y) / (1. + taut * torch.sqrt(u1x**2 + u1y**2 + 1e-12))
            p21 = (p21 + taut * u2x) / (1. + taut * torch.sqrt(u2x**2 + u2y**2 + 1e-12))
            p22 = (p22 + taut * u2y) / (1. + taut * torch.sqrt(u2x**2 + u2y**2 + 1e-12))
            del u1x
            del u1y
            del u2x
            del u2y
            
        return u1, u2
