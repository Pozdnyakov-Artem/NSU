import pandas as pd
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

df = pd.read_csv("production.csv")

liquid = df.groupby('API')[['Liquid', 'Gas', 'Water']].apply(lambda df_: df_.reset_index(drop=True))
df_prod = liquid.unstack()
# print(df_prod.head())

data = df_prod.values
# print(data.shape)
data = data.reshape(50,24,3)
data_max = data.max(axis=(0, 1), keepdims=True)
data = data / data_max
# data = data[:, :, np.newaxis]

data_tr = data[:40]
data_tst = data[40:]
# print(data_tr.shape)
x_data = [data_tr[:, i:i+12] for i in range(11)]
y_data = [data_tr[:, i+1:i+13] for i in range(11)]
#
x_data = np.concatenate(x_data, axis=0)
y_data = np.concatenate(y_data, axis=0)
# print(x_data.shape)
#
tensor_x = torch.Tensor(x_data) # transform to torch tensor
tensor_y = torch.Tensor(y_data)
#
oil_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
oil_dataloader = DataLoader(oil_dataset, batch_size=16) # create your dataloader
#
#
class OilModel(nn.Module):
    def __init__(self, timesteps=12, units=32):
        super().__init__()
        self.lstm1 = nn.LSTM(3, units, 2, batch_first=True)
        self.dense = nn.Linear(units, 3)
        self.relu = nn.ReLU()

    def forward(self, x):
        h, _ = self.lstm1(x)
        out = self.dense(h)
        out = self.relu(out)
        # outs = []
        # for i in range(h.shape[0]):
        #     outs.append(self.relu(self.dense(h[i])))
        # out = torch.stack(outs, dim=0)
        return out
#
model = OilModel()
opt = optim.Adam(model.parameters())
criterion = nn.MSELoss()
#
NUM_EPOCHS = 20
#
for epoch in range(NUM_EPOCHS):  # loop over the dataset multiple times
#
    running_loss = 0.0
    num = 0
    for x_t, y_t in oil_dataloader:
#         # zero the parameter gradients
        opt.zero_grad()
#
        # forward + backward + optimize
        outputs = model(x_t)
        loss = criterion(outputs, y_t)
        loss.backward()
        opt.step()
#
        # print statistics
        running_loss += loss.item()
        num += 1
#
    print(f'[Epoch: {epoch + 1:2d}] loss: {running_loss / num:.3f}')
#
print('Finished Training')

x_tst = data_tst[:, :12]
print(x_tst.shape)
predicts = np.zeros((x_tst.shape[0], 0, x_tst.shape[2]))
print(predicts.shape)

for i in range(12):
    x = np.concatenate((x_tst[:, i:], predicts), axis=1)
    x_t = torch.from_numpy(x).float()
    pred = model(x_t).detach().numpy()
    last_pred = pred[:, -1:, :]  # Нас интересует только последний месяц
    predicts = np.concatenate((predicts, last_pred), axis=1)
print(predicts.shape)
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
for iapi in range(4):
    plt.subplot(2, 2, iapi+1)
    plt.plot(np.arange(x_tst.shape[1]), x_tst[iapi, :, 0], label='Actual')
    plt.plot(np.arange(predicts.shape[1])+x_tst.shape[1], predicts[iapi, :, 0], label='Prediction')
    plt.legend()
plt.show()