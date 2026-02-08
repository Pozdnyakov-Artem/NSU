import numpy as np
import torchvision
import torch
from torch.utils.data import DataLoader

train_dataset = torchvision.datasets.MNIST(
    root="./MNIST/train", train=True,
    transform=torchvision.transforms.ToTensor(),
    download=False)

test_dataset = torchvision.datasets.MNIST(
    root="./MNIST/test", train=False,
    transform=torchvision.transforms.ToTensor(),
    download=False)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = torch.nn.Flatten()
        self.lin1 = torch.nn.Linear(28*28, 128)
        self.lin2 = torch.nn.Linear(128,64)
        self.lin3 = torch.nn.Linear(64,10)
        self.relu = torch.nn.ReLU()
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = self.flatten(x)
        x = self.relu(self.lin1(x))
        x = self.relu(self.lin2(x))
        # y = self.softmax(self.lin3(x))
        return x

    def fit(self, train_loader, epochs):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.001)
        l1_loss = torch.nn.CrossEntropyLoss()

        for epoch in range(epochs):

            for img, lbl in train_loader:
                optimizer.zero_grad()
                y_pred = self.forward(img)
                loss = l1_loss(y_pred, lbl)
                loss.backward()
                optimizer.step()

# img, lbl = map(torch.Tensor, zip(*train_dataset))
# print((train_dataset[0][1]))
# print(train_dataset.shape)
model = SimpleModel()
model.fit(train_loader, 5)
ans = model.forward(test_dataset[0][0])
print(torch.argmax(ans, dim=1), test_dataset[0][1])