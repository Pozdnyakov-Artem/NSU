import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from torch import nn
from torch.utils.data import Dataset
import torch.nn.functional as F
from torchvision.transforms import transforms

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = transforms.Compose([
    transforms.ToTensor(),
])

data = []

for i in range(1,41):
    for idx_first_img in range(1,10):
        for idx_second_img in range(idx_first_img+1,11):
            data.append((transform(cv2.imread(f"s{i}/{idx_first_img}.pgm",cv2.IMREAD_GRAYSCALE)),
                         transform(cv2.imread(f"s{i}/{idx_second_img}.pgm",cv2.IMREAD_GRAYSCALE)),1))

for idx_first_dir in range(1,41):
    for idx_first_img in range(1,11):
        for idx_second_dir in range(idx_first_dir+1,min(idx_first_dir+4,41)):
            for idx_second_img in range(6,11):
                data.append((transform(cv2.imread(f"s{idx_first_dir}/{idx_first_img}.pgm", cv2.IMREAD_GRAYSCALE)),
                             transform(cv2.imread(f"s{idx_second_dir}/{idx_second_img}.pgm", cv2.IMREAD_GRAYSCALE)), 0))

print(len(data))

class FaceDataset(Dataset):
    def __init__(self, img_data, transform=None):
        super().__init__()
        self.crops = img_data
        self.transform = transform

    def __getitem__(self, idx):
        img1, img2, label = self.crops[idx]

        return img1, img2, torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.crops)

dataset = FaceDataset(data)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

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
        # diff = abs(emb1 - emb2)
        # prob = self.classifier(diff)

        return emb1, emb2

base_model = model()
siamese = SiameseNet(base_model)
siamese.to(device)
class Contrasitive(nn.Module):
    def __init__(self,margin):
        super().__init__()
        self.margin = margin

    def forward(self,x1,x2,label):
        distance = F.pairwise_distance(x1,x2)
        loss = 0.5*label*torch.pow(distance,2)+ 0.5*(1-label)*torch.pow(torch.clamp(self.margin-distance,min=0),2)
        return loss.mean()

criterion = Contrasitive(1)
# criterion = nn.BCELoss()
optimizer = torch.optim.Adam(siamese.parameters(), lr=0.0005)

# for epoch in range(10):
#
#     running_loss = 0.0
#     num_batch = 0
#
#     for img1, img2, label in dataloader:
#         img1, img2, label = img1.to(device), img2.to(device), label.to(device)
#         optimizer.zero_grad()
#         out1,out2 = siamese(img1, img2)
#         loss = criterion(out1,out2, label)
#         loss.backward()
#         optimizer.step()
#
#         running_loss += loss.item()
#         num_batch += 1
#     print(f"Epoch {epoch}, Loss: {running_loss/num_batch}")

siamese.eval()

def test(img1, img2,i):
    with torch.no_grad():
        img_t1 = transform(cv2.imread(img1,cv2.IMREAD_GRAYSCALE)).unsqueeze(0).to(device)
        img_t2 = transform(cv2.imread(img2,cv2.IMREAD_GRAYSCALE)).unsqueeze(0).to(device)

        show_img = cv2.imread(img1,cv2.IMREAD_GRAYSCALE)
        show_img2 = cv2.imread(img2,cv2.IMREAD_GRAYSCALE)

        value = 1 - min(1,F.pairwise_distance(*siamese(img_t1,img_t2)).item())

        fig, axes = plt.subplots(1, 2, figsize=(8, 4))
        fig.suptitle(f"{value:.4f}", fontsize=14, fontweight='bold')

        axes[0].imshow(show_img, cmap='gray')
        axes[0].axis('off')
        axes[0].set_title("Image 1")

        axes[1].imshow(show_img2, cmap='gray')
        axes[1].axis('off')
        axes[1].set_title("Image 2")
        plt.tight_layout()
        plt.savefig(f"result{i}.png", dpi=300, bbox_inches='tight')

# img_t3 = transform(cv2.imread(r"s2/2.pgm",cv2.IMREAD_GRAYSCALE)).unsqueeze(0).to(device)

# with torch.no_grad():
#     print(1 - min(1,F.pairwise_distance(*siamese(img_t1,img_t2)).item()))
#     print(1 - min(1,F.pairwise_distance(*siamese(img_t1,img_t1)).item()))
#     print(1 - min(1,F.pairwise_distance(*siamese(img_t1,img_t3)).item()))
test(r"s1/1.pgm",r"s1/2.pgm",1)
test(r"s1/1.pgm",r"s1/1.pgm",2)
test(r"s1/1.pgm",r"s2/3.pgm",3)

with torch.no_grad():
    embeddings = []
    labels = []
    for i in range(1,5):
        for idx_first_img in range(1,10):
            item = transform(cv2.imread(f"s{i}/{idx_first_img}.pgm", cv2.IMREAD_GRAYSCALE)).unsqueeze(0).to(device)
            emb = siamese.base_model(item)
            embeddings.append(emb.squeeze(0).cpu().numpy())
            labels.append(i)

    tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
    embeddings_2d = tsne.fit_transform(np.array(embeddings))

    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1],
                          c=labels, cmap='tab20',
                          s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
    plt.colorbar(scatter, label="Person ID", ticks=sorted(np.unique(labels)))
    plt.tight_layout()
    plt.savefig("tsne_embeddings.png", dpi=300, bbox_inches='tight')