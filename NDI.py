import torch
import torch.fft as fft

class NDIErr(torch.nn.Module):
    def __init__(self):
        super(NDIErr, self).__init__()

    def forward(self, x):
        f1 = x[0]
        f2 = x[1]
        m  = x[2]
        f1d = torch.exp(torch.complex(torch.zeros(f1.shape, dtype = torch.float).to(device='cuda'), f1))
        f2d = torch.exp(torch.complex(torch.zeros(f2.shape, dtype = torch.float).to(device='cuda'), f2))

        err = torch.abs(torch.mul(m, (f1d-f2d)))     
        return err

class DoMask(torch.nn.Module):
    def __init__(self):
        super(DoMask, self).__init__()

    def forward(self, x):
        d = x[0]
        m = x[1]
             
        return d*m

