import torch
import torch.nn as nn

class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        self.inSize = input_size
        self.outSize=output_size