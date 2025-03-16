from torch.nn import Conv2d, Linear, BatchNorm2d
from torch import where, rand, topk, long, empty, zeros, no_grad, tensor
import math
from math import sqrt
import torch
import sys
from torch.nn.init import calculate_gain
import torch.nn.functional as F
import copy

def get_layer_bound(layer, init, gain):
    if isinstance(layer, Conv2d):
        return sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        if init == 'default':
            bound = sqrt(1 / layer.in_features)
        elif init == 'xavier':
            bound = gain * sqrt(6 / (layer.in_features + layer.out_features))
        elif init == 'lecun':
            bound = sqrt(3 / layer.in_features)
        else:
            bound = gain * sqrt(3 / layer.in_features)
        return bound


def get_layer_std(layer, gain):
    if isinstance(layer, Conv2d):
        return gain * sqrt(1 / (layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]))
    elif isinstance(layer, Linear):
        return gain * sqrt(1 / layer.in_features)
    
class HiddenImp(object):
    def __init__(self, net, hidden_activation, device=torch.device("cpu")):
        super(HiddenImp, self).__init__()

        self.net = net
        self.bn_layers = []
        self.weight_layers = []
        self.get_weight_layers(nn_module=self.net)
        self.num_hidden_layers = int(len(self.weight_layers) / 2)
        self.device = device

        self.util,  self.mean_feature_mag = [], []

        for i in range(self.num_hidden_layers):
            self.util.append(zeros(self.weight_layers[2*i].out_features, dtype=torch.float32, device=self.device))
            self.mean_feature_mag.append(zeros(self.weight_layers[2*i].out_features, dtype=torch.float32, device=self.device))

        self.accumulated_num_features_to_replace = [0 for i in range(self.num_hidden_layers)]
        self.m = torch.nn.Softmax(dim=1)


        self.stds = self.compute_std(hidden_activation=hidden_activation)
 
        self.num_new_features_to_replace = []  
        for i in range(self.num_hidden_layers):
            with no_grad():
                self.num_new_features_to_replace.append(self.replacement_rate * self.weight_layers[2*i].out_features)  

    def get_weight_layers(self, nn_module: torch.nn.Module):
        if isinstance(nn_module, Conv2d) or isinstance(nn_module, Linear):
            self.weight_layers.append(nn_module)
        elif isinstance(nn_module, BatchNorm2d):
            self.bn_layers.append(nn_module)
        else:
            for m in nn_module.children():
                if hasattr(nn_module, 'downsample'):
                    if nn_module.downsample == m:   continue
                self.get_weight_layers(nn_module=m)

    def compute_std(self, hidden_activation):
        stds = []
        gain = calculate_gain(nonlinearity=hidden_activation)
        for i in range(self.num_hidden_layers):
            stds.append(get_layer_std(layer=self.weight_layers[i], gain=gain))
        stds.append(get_layer_std(layer=self.weight_layers[-1], gain=1))
        return stds
    
    def obain_imp(self, features):
        for i in range(0, self.num_hidden_layers):
            with torch.no_grad():
                feature_tmp = (1 - self.decay_rate) * features[i].abs().mean(dim=0) 
                sim = F.cosine_similarity(feature_tmp, feature_tmp[0], dim=1)
                sim = (sim-sim.min())/(sim.max() - sim.min() + 1e-8)
                feature_tmp = feature_tmp * sim.unsqueeze(1) 
                feature_tmp = feature_tmp.mean(dim=0)
                self.mean_feature_mag[i] += feature_tmp  
                next_layer = self.weight_layers[i*2 + 1]
                if isinstance(next_layer, Linear):
                    output_wight_mag = next_layer.weight.data.abs().mean(dim=0) 
                self.util[i] += output_wight_mag