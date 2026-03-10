import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn, optim
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])

trainset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)
testset = torchvision.datasets.FashionMNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)


class AutoEncoder(nn.Module):
    def __init__(self, img_shape, inp_dim, hidden_dim):
        super(AutoEncoder, self).__init__()
        self.img_shape = img_shape
        self.encoder = nn.Sequential(nn.Linear(inp_dim, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, hidden_dim)
                                     )
        self.decoder = nn.Sequential(nn.Linear(hidden_dim, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 256),
                                     nn.ReLU(),
                                     nn.Linear(256, inp_dim)
                                     )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.encoder(x)
        x = self.decoder(x)
        return x.view(x.size(0),*self.img_shape)

    def encode(self, x):
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self,x):
        return self.decoder(x).view(*self.img_shape)

class ConvAutoEncoder(nn.Module):
    def __init__(self, img_shape, inp_dim, hidden_dim):
        super().__init__()
        self.encoder=nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),  # -> 16x28x28
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # -> 16x14x14
            nn.Conv2d(16, 32, kernel_size=3, padding=1),  # -> 32x14x14
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )

        self.decoder=nn.Sequential(
            nn.ConvTranspose2d(32,16,kernel_size=3, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(16,1,kernel_size=2, stride=2),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)


NUM_EPOCHS = 20
LEARNING_RATE = 1e-3
BATCH_SIZE = 128

trainloader = DataLoader(
    trainset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True
)
testloader = DataLoader(
    testset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=32,
    pin_memory=True,
    persistent_workers=True
)
def train(model, trainloader, index):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    arr_loss=[]

    for epoch in range(NUM_EPOCHS):
        train_loss = 0
        counter = 0
        for images, _ in trainloader:
            images = images.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, images)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * images.size(0)
            counter+=images.size(0)
        arr_loss.append(train_loss)
        print(f"epoch {epoch}: loss {train_loss/counter}")
    torch.save(model.state_dict(), f"model{index}.pth")
    return arr_loss

def count_matches(start, end):
    diff = torch.abs(end - start)
    max_error_per_image = diff.flatten(start_dim=1).max(dim=1).values
    matches = (max_error_per_image <= 0.01)

    return matches.sum().item()

def val(vers, index):

    if vers == 'conv':
        model = ConvAutoEncoder([28,28],28*28,28).to(device)
    else:
        model = AutoEncoder([28,28],28*28,28).to(device)

    model.load_state_dict(torch.load(f"model{index}.pth", map_location=device))
    with torch.no_grad():
        ans = 0
        vsego = 0
        model.eval()
        for images, _ in testloader:
            images = images.to(device, non_blocking=True)
            outputs = model(images)
            ans+=count_matches(images, outputs)
            vsego+=images.size(0)

        print(f"точность {ans/vsego}")
model = AutoEncoder([28,28],28*28,28).to(device)
fc_loss = train(model, trainloader,1)
model = ConvAutoEncoder([28,28],28*28,28).to(device)
conv_loss = train(model, trainloader,2)

val('lin',1)
val('conv',2)

plt.figure(figsize=(10,10))
plt.plot(list(range(1,20)),fc_loss,label='fc_loss',color='red')
plt.plot(list(range(1,20)),conv_loss,label='conv_loss',color='blue')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.savefig('graph.png')


def plot_decoder_grid(model, index, latent_dim=2, range_val=3, n_samples=15):
    model.eval()
    device = next(model.parameters()).device

    x = np.linspace(-range_val, range_val, n_samples)
    y = np.linspace(-range_val, range_val, n_samples)
    xx, yy = np.meshgrid(x, y)
    coords = np.stack([xx.ravel(), yy.ravel()], axis=1)

    with torch.no_grad():
        latent = torch.FloatTensor(coords).to(device)

        decoded = model.decode(latent)
        decoded = decoded.cpu().clamp(0, 1)

    fig, axes = plt.subplots(n_samples, n_samples, figsize=(10, 10))
    for i, ax in enumerate(axes.flat):
        if i < len(decoded):
            ax.imshow(decoded[i].squeeze(), cmap='gray')
        ax.axis('off')

        plt.suptitle('Decoder Output over Latent Space', y=1.02)
        plt.tight_layout()
        # plt.show()
    plt.savefig(f'decoder{index}.png')

plot_decoder_grid(AutoEncoder([28,28],28*28,28),1)
plot_decoder_grid(ConvAutoEncoder([28,28],28*28,28),2)