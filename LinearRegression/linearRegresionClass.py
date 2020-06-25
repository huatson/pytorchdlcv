import torch
import torch.nn as nn


class LinearRegression(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.inSize = input_size
        self.outSize = output_size
        self.linearModel = nn.Linear(in_features=self.inSize, out_features=self.outSize)

    def forward(self, x):
        prediction = self.linearModel(x)
        return prediction


def main():
    # set custom seed
    torch.manual_seed(1)
    # create linear model
    lModel = LinearRegression(1, 1)
    # test sample model
    listdata = list(lModel.parameters())
    print(listdata)
    # make prediction
    x = torch.tensor([[1.0], [2.0]])
    pred = lModel.forward(x)
    print(pred)


if __name__ == "__main__":
    main()
