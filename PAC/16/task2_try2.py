import re
import gensim
import numpy as np
import torch
from torch import nn, optim

text = open("text.txt", "r", encoding='utf-8')

sentences = re.split(r'(?<=[.!?])\s+', text.read())
pattern = re.compile(r"\w+")

sentences = [pattern.findall(sentence.lower()) for sentence in sentences]

word_model = gensim.models.Word2Vec(sentences,
                                    vector_size=100,
                                    min_count=1,
                                    window=5,
                                    epochs=100)

word_to_idx = {word: idx for idx, word in enumerate(word_model.wv.index_to_key)}
idx_to_word = {idx: word for word, idx in word_to_idx.items()}

pretrained_weights = word_model.wv.vectors
vocab_size, embedding_size = pretrained_weights.shape

def create_data(wind_size=5):
    x, y = [], []
    for sentence in sentences:
        if len(sentence) <= wind_size:
            continue
        for i in range(len(sentence) - wind_size):
            window = sentence[i:wind_size + i]
            target = sentence[i + wind_size]

            x.append([word_model.wv[word] for word in window])
            y.append(word_to_idx[target])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int64)


data_x, data_y = create_data()

x_tensor = torch.tensor(data_x, dtype=torch.float32)
y_tensor = torch.tensor(data_y, dtype=torch.long)

dataset = torch.utils.data.TensorDataset(x_tensor, y_tensor)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

class RNN(nn.Module):
    def __init__(self, emb_size, hidden_size=128, num_layers=1, vocab_size=None):
        super().__init__()
        self.lstm = nn.LSTM(emb_size, hidden_size, num_layers, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        output, hidden = self.lstm(x, hidden)
        output = self.linear(output)
        return output, hidden


model = RNN(emb_size=100, hidden_size=128, num_layers=2, vocab_size=vocab_size)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

print("\n=== Начало обучения ===")
for epoch in range(50):
    model.train()
    total_loss = 0

    for x_batch, y_batch in dataloader:
        optimizer.zero_grad()
        output, _ = model(x_batch)
        output = output[:, -1, :]
        loss = criterion(output, y_batch)
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        total_loss += loss.item()

    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1:2d}/{50} | Loss: {total_loss / len(dataloader):.4f}")


def generate_text(model, start_words, length=5, temperature=0.8, window_size=5):
    model.eval()

    word_vectors = [word_model.wv[word] for word in start_words]
    generated = start_words.copy()

    for _ in range(length):
        window = word_vectors[-window_size:]
        window_np = np.array(window, dtype=np.float32)
        x = torch.from_numpy(window_np).unsqueeze(0)

        with torch.no_grad():
            output, _ = model(x, hidden=None)

            if output.dim() == 3:
                logits = output[:, -1, :]
            else:
                logits = output[-1:].unsqueeze(0)

        output_dist = torch.softmax(logits[0] / temperature, dim=0)
        top_i = torch.multinomial(output_dist, 1).item()

        next_word = idx_to_word[top_i]
        generated.append(next_word)

        word_vectors.append(word_model.wv[next_word])

    return ' '.join(generated)

print("\n=== Генерация текста ===")
print(generate_text(model, ["небольшая", "комната", "в", "которую", "прошел"], temperature=0.5))
print(generate_text(model, ["он", "даже", "знал", "сколько", "шагов"], temperature=0.6))
print(generate_text(model, ["старуха", "полезла", "в", "карман", "за"], temperature=0.7))