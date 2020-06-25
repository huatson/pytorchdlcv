import torch
from torch.nn import Linear


def forward(w, x, b):
    return (w*x)+b


def SimpleLinearModel():
    # test torch tensors
    w = torch.tensor(3.0, requires_grad=True)
    b = torch.tensor(1.0, requires_grad=True)
    x = torch.tensor(2)
    y = forward(w, x, b)
    print(y)
    x = torch.tensor([[4], [7]])
    y = forward(w, x, b)
    print(y)

    # start random seed
    torch.manual_seed(1)

    # build linear model
    model = Linear(in_features=1, out_features=1)
    print(model.bias, model.weight)

    # make prediction 1
    x = torch.tensor([2.0])
    print(model(x))

    # make prediction 2
    x = torch.tensor([[2.0], [3.3]])
    print(model(x))


def main():
    '''
    Linear model (weight and bias)
    y=w*x+b
    '''
    SimpleLinearModel()


if __name__ == "__main__":
    main()
