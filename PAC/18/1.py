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
# print(transform(cv2.imread("s1/1.pgm", cv2.IMREAD_GRAYSCALE)).shape)
for i in range(1,41):
    for idx_first_img in range(1,10):
        for idx_second_img in range(i+1,11):
            data.append((transform(cv2.imread(f"s{i}/{idx_first_img}.pgm",cv2.IMREAD_GRAYSCALE)),
                         transform(cv2.imread(f"s{i}/{idx_second_img}.pgm",cv2.IMREAD_GRAYSCALE)),1))

for idx_first_dir in range(1,41):
    for idx_first_img in range(1,11):
        for idx_second_dir in range(idx_first_dir+1,41):
            for idx_second_img in range(1,11):
                data.append((transform(cv2.imread(f"s{idx_first_dir}/{idx_first_img}.pgm", cv2.IMREAD_GRAYSCALE)),
                             transform(cv2.imread(f"s{idx_second_dir}/{idx_second_img}.pgm", cv2.IMREAD_GRAYSCALE)), 0))

# print(len(data))

class FaceDataset(Dataset):
    def __init__(self, img_data, transform=None):
        super().__init__()
        # self.crops = self.get_crops(img_data)
        self.crops = img_data
        self.transform = transform

    def __getitem__(self, idx):
        img1, img2, label = self.crops[idx]

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.crops)

dataset = FaceDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, shuffle=True)

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
        # self.classifier = nn.Sequential(
        #     nn.Linear(128, 1),
        #     nn.Sigmoid()
        # )

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
        loss = label*torch.pow(distance,2)+ (1-label)*torch.pow(torch.clamp(self.margin-distance,min=0),2)

        # loss_correct = label * torch.pow(distance, 2) + \
        #                (1 - label) * torch.pow(torch.clamp(self.margin - distance, min=0), 2)
        return loss.mean()

criterion = Contrasitive(1)
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(siamese.parameters(), lr=0.0005)

for epoch in range(20):

    running_loss = 0.0
    num_batch = 0

    for img1, img2, label in dataloader:
        optimizer.zero_grad()
        # print(label)
        # out = siamese(img1, img2)
        out1 = siamese.base_model(img1)
        out2 = siamese.base_model(img2)
        loss = criterion(out1,out2, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        num_batch += 1
    print(f"Epoch {epoch}, Loss: {running_loss/num_batch}")

siamese.eval()

img_t1 = transform(cv2.imread(r"s1/1.pgm",cv2.IMREAD_GRAYSCALE)).unsqueeze(1)
img_t2 = transform(cv2.imread(r"s1/2.pgm",cv2.IMREAD_GRAYSCALE)).unsqueeze(1)
img_t3 = transform(cv2.imread(r"s2/2.pgm",cv2.IMREAD_GRAYSCALE)).unsqueeze(1)
# print(img_t1.shape)
with torch.no_grad():
    print(F.pairwise_distance(siamese.base_model(img_t1),siamese.base_model(img_t2)).item())
    print(F.pairwise_distance(siamese.base_model(img_t1),siamese.base_model(img_t1)).item())
    print(F.pairwise_distance(siamese.base_model(img_t1),siamese.base_model(img_t3)).item())
