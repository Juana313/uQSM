import torch


def gradient(x):
    dx = (x[:, :, :-1, :-1, :-1] - x[:, :, 1:, :-1, :-1])
    dy = (x[:, :, :-1, :-1, :-1] - x[:, :, :-1, 1:, :-1])
    dz = (x[:, :, :-1, :-1, :-1] - x[:, :, :-1, :-1, 1:])
    
    return dx, dy, dz

def tv_loss(y_pred):
    dx, dy, dz = gradient(y_pred)
    tv_loss = torch.mean(torch.abs(dx)) + \
              torch.mean(torch.abs(dy)) + \
              torch.mean(torch.abs(dz))
    return tv_loss