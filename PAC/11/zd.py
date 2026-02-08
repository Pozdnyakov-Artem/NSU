import torch


def create_model(in_ch, out_ch):
    model = torch.nn.Sequential(
        torch.nn.Linear(in_ch, 32),
        torch.nn.ReLU(),
        torch.nn.Linear(32, out_ch),
        torch.nn.ReLU(),
    )

    return model

def fit(model, train_loader, epochs, lr):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    l1_loss = torch.nn.L1Loss()
    for epoch in range(epochs):
        for x, y in train_loader:
            optimizer.zero_grad()
            y_pred = model(x)
            loss = l1_loss(y_pred, y)
            loss.backward()
            optimizer.step()


def create_model2():
    model = torch.nn.Sequential(
        torch.nn.Linear(256, 64),
        torch.nn.ReLU(),
        torch.nn.Linear(64, 16),
        torch.nn.Tanh(),
        torch.nn.Linear(16, 4),
        torch.nn.Softmax(dim=1),
    )

    return model


class model3(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3, out_channels=8, kernel_size=3, stride=1, padding=1)
        self.relu1 = torch.nn.ReLU()
        self.pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = torch.nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, stride=1, padding=1)
        self.relu2 = torch.nn.ReLU()
        self.pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2)

    def forward(self,x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)

        return x

test = torch.rand(4,3, 19, 19)
model = model3()
print(test.shape,model(test).shape)

flat = torch.nn.Flatten()
sl1 = create_model(19*19*3,256)
sl2 = create_model2()
sl11 = model3()

comb_mod = torch.nn.Sequential(
    # flat,
    # sl1,
    # sl11,
    model3(),
    flat,
    sl2
)

print(comb_mod(test).shape)