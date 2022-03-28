import torch
import random
import struct
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
from typing import Union


class IsEvenNN(object):
    def __init__(self, optimizer=torch.optim.Adam, criterion=nn.L1Loss()):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.net = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 16),
            nn.ReLU(),
            nn.Linear(16, 1),
            nn.Sigmoid()
        )
        self.net.to(self.device)
        self.optimizer = optimizer(self.net.parameters())
        self.criterion = criterion

    def train(self, xtrain: list, ytrain: list, n_epochs: int):
        assert len(xtrain) == len(ytrain), "data and label list are not same length"
        train_set = TensorDataset(torch.tensor(xtrain, device=self.device), torch.tensor(ytrain, device=self.device))
        self.net.train()
        for epoch in range(n_epochs):  # loop over the dataset multiple times

            running_loss = 0.0
            for i, (inputs, labels) in enumerate(DataLoader(train_set, batch_size=32)):

                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.net(inputs)
                loss = self.criterion(outputs, labels.unsqueeze(1))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()
                if i % 2000 == 1999:  # print every 2000 mini-batches
                    print(f'[{epoch}, {i + 1:5d}] loss: {running_loss / 2000:.3e}')
                    running_loss = 0.0

        print('Finished Training')

    def predict(self, xtest: list[list[float]], extras=False):
        self.net.eval()
        xtest_tensor = torch.tensor(xtest, device=self.device)
        with torch.no_grad():
            outputs = self.net(xtest_tensor).squeeze().tolist()  # float outputs of the network
            if type(outputs) is not list:
                outputs = [outputs]
            predictions = [y > 0.5 for y in outputs]  # boolean predictions
            return predictions, outputs if extras else predictions

    def accuracy(self, xtest: list[list[float]], ytest: list[Union[float, bool]]) -> float:
        preds = self.predict(xtest)
        if type(ytest) == list[float]:
            ytest = [y > 0.5 for y in ytest]  # convert ground truths to boolean
        corrects = [x == y for x, y in zip(preds, ytest)]
        return sum(corrects)/len(corrects)

    def predict_single(self, number) -> tuple[bool, float]:
        """Predict a single number. Any format that can be understood by int() is accepted."""
        bits = binary_int(int(number))
        if len(bits) != 32:
            raise IndexError(f"Could not convert {number} to 32-bit int")
        outputs, conf = self.predict([bits], extras=True)
        return outputs[0], conf[0]

    def __call__(self, *args, **kwargs):
        return self.net(*args, **kwargs)


def _random_float(lower, upper) -> float:
    return random.random() * (upper - lower) + lower


def binary_float(num: float, network=True) -> list[float]:
    """Convert a float to a 32-long list of bits according to IEEE 754.

    :param: num number to be converted, must be float
    :param: network format in network byte order, i.e. big endian. Default True.
    :returns: list of float, either 1.0f or 0.0f (this is because Pytorch uses float tensors)
    """
    fmt = '!f' if network else 'f'
    bitstring = ''.join(bin(c).replace('0b', '').rjust(8, '0') for c in struct.pack(fmt, num))
    return [float(bit) for bit in bitstring]


def binary_int(num: int) -> list[float]:
    bitstring = format(num, 'b').rjust(32, '0')
    return [float(bit) for bit in bitstring]


def generate_floats(length: int, lower: float = 0, upper: float = 1e9) -> tuple[
    list[float], list[list[float]], list[float]]:
    xarray = []
    yarray = []
    floats = []
    for i in range(length):
        number = _random_float(lower, upper)
        floats.append(number)
        xarray.append(binary_float(number))
        yarray.append(float(int(number) % 2 == 0))
    return floats, xarray, yarray


def generate_ints(length: int, lower: int = 0, upper: int = 0xfffffff) -> tuple[
    list[int], list[list[float]], list[float]]:
    xarray = []
    yarray = []
    ints = []
    for i in range(length):
        number = random.randint(lower, upper)
        ints.append(number)
        xarray.append(binary_int(number))
        yarray.append(float(number % 2 == 0))
    return ints, xarray, yarray
