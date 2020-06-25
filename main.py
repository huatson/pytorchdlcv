import time
import os
import sys
import torch
import numpy as np
from matplotlib import pyplot as plt


# Import apex's distributed module.
try:
    from apex.parallel import DistributedDataParallel
except ImportError:
    raise ImportError("can not load APEX")


def DimensionalTensors():
    '''
    https://pytorch.org/docs/stable/tensors.html
    '''
    # create tensors
    v = torch.tensor([1, 2, 3, 4, 5, 6])
    f = torch.FloatTensor([1, 2, 3, 4, 5, 6])

    # reshape rows columns
    x = v.view(3, -1)
    print(x)

    # from numpy to tensor
    numpytensor = np.array([1, 2, 3, 4, 5, 6])
    torchtensor = torch.from_numpy(numpytensor)
    print(torchtensor, torchtensor.type())

    numpytensor = torchtensor.numpy()
    print(numpytensor)


def VectorOps():
    t1 = torch.tensor([1, 2, 3])
    t2 = torch.tensor([1, 2, 3])
    print(t1+t2)
    print(t1*t2)
    print(t1*5)
    dotprod = torch.dot(t1, t2)
    print(dotprod)

    x = torch.linspace(0, 10, 5)
    print(x)
    y = torch.exp(x)
    z = torch.sin(x)
    plt.plot(x.numpy(), z.numpy())
    plt.show()


def DimensionalTensors2():
    oneDimension = torch.arange(0,9)
    print(oneDimension)
    twoDimension = oneDimension.view(3,3)
    print(twoDimension)
    #access values
    print(twoDimension[0,0]) #  0
    print(twoDimension[1,2]) #  5

    x = torch.arange(18).view(2,3,3)
    print(x)
    print(x.view(3,3,2))



def main():
    print("init")
    # DimensionalTensors()
    #VectorOps()
    DimensionalTensors2()


if __name__ == "__main__":
    main()
