import torch.nn as nn

def activation(activation_type='relu', **kwargs):
    if activation_type == 'relu':
        return nn.ReLU(inplace=True)
    elif activation_type == 'prelu':
        return nn.PReLU()
    elif activation_type == 'leaky_relu':
        return nn.LeakyReLU(inplace=True)
    elif activation_type == 'tanh':
        return nn.Tanh()
    else:
        raise ValueError('invalid activation type: {}'.format(activation_type))

