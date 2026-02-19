import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import TensorDataset, DataLoader

file = open("prepared_text.txt", "r", encoding="utf-8")

sentences = [sentence for sentence in file.read().split('.') if len(sentence) > 0]

dic = set()

for sentence in sentences:
    for word in sentence.lower().split():
        for letter in word:
            dic.add(letter)

sl = list(dic)
sl.append(' ')

data = []

for sentence in sentences:
    emb_sentence = []
    for word in sentence.lower().split():
        for letter in word:
            emb_word=[0]*len(sl)
            emb_word[sl.index(letter)]=1
            emb_sentence.append(emb_word)
        emb_sentence.append(([0]*(len(sl)-1))+[1])
    data.append(emb_sentence[:31])

train_data = []
label = []

for sentence in data:
    train_data.append(sentence[:-1])
    label.append(sentence[1:])

X = torch.tensor(np.array(train_data, dtype=np.float32))
y = torch.tensor(np.array(label, dtype=np.float32))

train_dataset = TensorDataset(X, y)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size = 128, num_layers=1):
        super().__init__()
        self.lstm1 = nn.LSTM(input_size,  #размер входного вектора
                          hidden_size, #число нейронов
                          num_layers,  #количество слоёв
                          batch_first=True)
        self.linear = nn.Linear(hidden_size, input_size)

    def forward(self,x, hidden = None):
        output, hidden = self.lstm1(x, hidden)
        output = self.linear(output)
        return output, hidden

    @torch.no_grad()
    def generate(self, st, length, temperature = 0.6):
        self.eval()

        st_tensor = []
        gen = st.lower()
        for let in gen:
            vec = [0]*len(sl)
            vec[sl.index(let)] = 1
            st_tensor.append(vec)

        if len(st_tensor) < X.shape[1]:
            pad_vec = [0] * len(sl)
            pad_vec[sl.index(' ')] = 1
            st_tensor = [pad_vec] * (X.shape[1] - len(st_tensor)) + st_tensor

        inp_seq = torch.tensor([st_tensor], dtype=torch.float32)
        hidden = None

        for _ in range(length):
            output, hidden = self(inp_seq, hidden)

            logits = output[0, -1, :]

            logits = logits / temperature
            probs = torch.softmax(logits, dim=0).numpy()
            next_idx = np.random.choice(len(sl), p=probs)
            next_char = sl[next_idx]
            gen += next_char

            new_vec = [0] * len(sl)
            new_vec[next_idx] = 1
            inp_seq = torch.cat([
                inp_seq[:, 1:, :],
                torch.tensor([[new_vec]], dtype=torch.float32)
            ], dim=1)

        return gen


model = RNN(len(sl), 128)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.003)

num_epochs = 100
best_val_loss = float('inf')

for epoch in range(num_epochs):

    model.train()
    train_loss = 0.0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X, batch_y

        optimizer.zero_grad()

        batch_y_indices = batch_y.argmax(dim=2)

        output, _ = model(batch_X)
        output_reshaped = output.view(-1, len(sl))
        y_reshaped = batch_y_indices.view(-1)

        loss = criterion(output_reshaped, y_reshaped)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    train_loss /= len(train_loader)


print(model.generate("солнце светит ярко над тихой р",5))