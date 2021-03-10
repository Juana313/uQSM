import torch
import torch.fft as fft

class CalFMLayer(torch.nn.Module):
    def __init__(self):
        super(CalFMLayer, self).__init__()

    def forward(self, inputs):
        suscp = inputs[0]
        kernel = inputs[1]
        
        ks = fft.fftn(suscp, dim=[-3, -2, -1])
                
        ks = ks*kernel
        fm = torch.real(fft.ifftn(ks, dim=[-3, -2, -1])) 
        
        return fm


