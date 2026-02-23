import cv2
import torch
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import transforms

transform = transforms.Compose([
    transforms.ToTensor(),
])

data = []
print(transform(cv2.imread("s1/1.pgm", cv2.IMREAD_GRAYSCALE)).shape)
for i in range(1,41):
    for idx_first_img in range(1,10):
        for idx_second_img in range(i+1,11):
            data.append((transform(cv2.imread(f"s{i}/{idx_first_img}.pgm",cv2.IMREAD_GRAYSCALE)),transform(cv2.imread(f"s{i}/{idx_second_img}.pgm",cv2.IMREAD_GRAYSCALE)),1))
            data.append((transform(cv2.imread(f"s{i}/{idx_first_img}.pgm",cv2.IMREAD_GRAYSCALE)), transform(cv2.imread(f"s{idx_second_img}/{i}.pgm",cv2.IMREAD_GRAYSCALE)), 0))

# print(len(data))

class FaceDataset(Dataset):
    def __init__(self, img_data, transform=None):
        super().__init__()
        # self.crops = self.get_crops(img_data)
        self.crops = img_data
        self.transform = transform

    def get_crops(self, data):
        return data

    def __getitem__(self, idx):
        img1, img2, label = self.crops[idx]

        if self.transform:
            if len(img1.shape) == 2:
                img1 = img1[:, :, None]
                img2 = img2[:, :, None]
            input_img1 = torch.from_numpy(img1).float() / 255.0
            input_img2 = torch.from_numpy(img2).float() / 255.0

            if input_img1.dim() == 2:
                input_img1 = input_img1.unsqueeze(0)
                input_img2 = input_img2.unsqueeze(0)
        else:
            input_img1 = img1
            input_img2 = img2

        return input_img1, input_img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.crops)

dataset = FaceDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=4, shuffle=True)

class model(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 96, 11),
            nn.ReLU(),
            nn.BatchNorm2d(96),
            nn.MaxPool2d(2, 2),

            nn.Conv2d(96, 256, 5),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(2),

            nn.Conv2d(256, 384, 3),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.2)
        )
        self.adaptive_avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc1 = nn.Linear(256,1024)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(1024,128)

    def forward(self,x):
        out = self.features(x)
        out = self.adaptive_avg_pool(out)
        out = torch.flatten(out, 1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

class SiameseNet(nn.Module):
    def __init__(self,base_model):
        super().__init__()
        self.base_model = base_model
        self.classifier = nn.Sequential(
            nn.Linear(128, 1),  # Выход - 1 число
            nn.Sigmoid()  # Превращаем в вероятность от 0 до 1
        )

    def forward(self,x1,x2):
        emb1 = self.base_model(x1)
        emb2 = self.base_model(x2)

        diff = abs(emb1 - emb2)

        prob = self.classifier(diff)

        return prob

base_model = model()
siamese = SiameseNet(base_model)

class Contrasitive(nn.Module):
    def __init__(self,margin):
        super().__init__()
        self.margin = margin

    def forward(self,x1,x2,label):
        distance = F.pairwise_distance(x1,x2)
        loss = (1-label)*torch.pow(distance,2)+ label*torch.pow(torch.clamp(self.margin-distance,min=0),2)

        loss_correct = label * torch.pow(distance, 2) + \
                       (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0), 2)
        return loss_correct.mean()

# criterion = Contrasitive(1)
criterion = nn.BCELoss()
optimizer = torch.optim.Adam(siamese.parameters(), lr=0.001)

for epoch in range(5):
    for img1, img2, label in dataloader:
        optimizer.zero_grad()
        # print(label)
        out = siamese(img1, img2)
        loss = criterion(out, label.unsqueeze(1))
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

siamese.eval()

img_t1 = transform(cv2.imread(r"s1/1.pgm",cv2.IMREAD_GRAYSCALE))
img_t2 = transform(cv2.imread(r"s1/2.pgm",cv2.IMREAD_GRAYSCALE))

print(siamese(img_t1,img_t2))