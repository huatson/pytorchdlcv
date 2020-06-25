import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.inSize = input_size
        self.outSize = output_size
        self.linearModel = nn.Linear(
            in_features=self.inSize, out_features=self.outSize)

    def forward(self, x):
        prediction = self.linearModel(x)
        return prediction


def getModelParameters(model):
    """
    return weight and bias (b0, b1) linear parameters
    """
    [w, b] = model.parameters()
    # due w is a 2d data, access to each one
    w1 = w[0][0].item()
    b1 = b[0].item()
    print("linear parameters w:{:.5f} b:{:.5f}".format(w1, b1))
    return w1, b1


def generateData(samples, constant, sd, printValues=False):
    '''
    randn:
        Returns a tensor filled with random numbers from a normal distribution 
        with mean 0 and variance 1 (also called the standard normal distribution).
        normally distributed shape: rows (sample points), cols (# of values for each point = 1)
        out_i ∼N(0,1)
    '''
    X_data = torch.randn(samples, 1)*constant
    noise = torch.randn(samples, 1)*sd
    y_value = X_data+noise
    if printValues:
        plt.plot(X_data.numpy(), y_value.numpy(), '*')
        plt.ylabel('y')
        plt.xlabel('x')
        plt.show()
    return X_data, y_value


def plotFittedModel(plotTitle, model, xData, yValues):
    """
    Plot data with current linear model
    """
    plt.title = plotTitle
    [w, b] = getModelParameters(model)
    x = np.array([-34, 30])
    y = (w*x)+b
    plt.plot(x, y, 'r')
    plt.scatter(xData, yValues)
    plt.show()


def SimpleModel():
    # set custom seed
    torch.manual_seed(1)
    # create dataset
    X, y = generateData(100, 10, 3, True)
    # create linear model
    lModel = LinearRegression(1, 1)
    plotFittedModel("regression model", lModel, X, y)


def main():
    # create simple model
    SimpleModel()


if __name__ == "__main__":
    main()
