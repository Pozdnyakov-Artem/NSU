import os

import cv2
import numpy as np
import torch
import torchvision
from PIL import Image
from matplotlib import pyplot as plt
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import transforms, models

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    temp_dataset = torchvision.datasets.ImageFolder(root = "2750")

    train_idx, test_idx = train_test_split(
                                range(len(temp_dataset)),
                                        test_size=0.2,
                                        random_state=42,)

    train_dataset = torchvision.datasets.ImageFolder(root = "2750",transform=transform)
    test_dataset = torchvision.datasets.ImageFolder(root = "2750",transform=transform)

    train_dataset = Subset(train_dataset, train_idx)
    test_dataset = Subset(test_dataset, test_idx)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)

    class L2Norm(nn.Module):
        def __init__(self):
            super().__init__()

        def forward(self, x):
            return F.normalize(x, p=2, dim=1)

    model = torchvision.models.resnet18(weights=models.ResNet18_Weights.DEFAULT)

    # model.fc = torch.nn.Linear(model.fc.in_features, 10)
    new_fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        L2Norm(),
    )
    model.fc = new_fc

    class ArcFaceHead(nn.Module):
        def __init__(self, in_features, out_features, s=64.0, m=0.50):
            super().__init__()
            self.s = s
            self.m = m
            self.weight = nn.Parameter(torch.FloatTensor(out_features, in_features))
            nn.init.xavier_uniform_(self.weight)
            self.register_buffer('cos_m', torch.tensor(np.cos(m), dtype=torch.float32))
            self.register_buffer('sin_m', torch.tensor(np.sin(m), dtype=torch.float32))
            self.register_buffer('th', torch.tensor(np.cos(np.pi - m), dtype=torch.float32))
            self.register_buffer('mm', torch.tensor(np.sin(np.pi - m) * m, dtype=torch.float32))

        def forward(self, x, label=None):

            w = F.normalize(self.weight, p=2, dim=1)

            cosine = F.linear(x, w)

            cosine_clamped = torch.clamp(cosine, -1.0, 1.0)

            theta = torch.acos(cosine_clamped)

            theta_plus_m = theta + self.m

            phi = torch.cos(theta_plus_m)

            one_hot = F.one_hot(label, num_classes=self.weight.shape[0]).float()
            output = (one_hot * phi) + ((1.0 - one_hot) * cosine)

            return output * self.s

        # def forward(self, x, label=None):
        #     # x = F.normalize(x, p=2, dim=1)
        #     w = F.normalize(self.weight, p=2, dim=1)
        #     cosine = F.linear(x, w)
        #
        #     sine = torch.sqrt(1.0 - torch.pow(cosine, 2))
        #     phi = cosine * self.cos_m - sine * self.sin_m
        #     phi = torch.where(cosine > self.th, phi, cosine - self.mm)
        #
        #     one_hot = F.one_hot(label, num_classes=self.weight.shape[0]).float()
        #     output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        #     return output * self.s


    archead = ArcFaceHead(512,10).to(device)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(list(model.parameters()) + list(archead.parameters()), lr=0.001)

    model.train()

    for epoch in range(40):
        num_batch = 0
        train_loss = 0
        for x_train, y_train in train_loader:
            x_train, y_train = x_train.to(device), y_train.to(device)
            optimizer.zero_grad()
            output = model(x_train)
            output = archead(output, y_train)

            loss = criterion(output,y_train)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            num_batch += 1
        print(train_loss/num_batch)

    model.eval()
    archead.eval()

    data = []
    labels = []

    def test(img1, img2, i):
        with torch.no_grad():
            img_t1 = Image.open(img1).convert('RGB')
            img_t2 = Image.open(img2).convert('RGB')

            img_t1 = transform(img_t1).unsqueeze(0).to(device)
            img_t2 = transform(img_t2).unsqueeze(0).to(device)

            show_img1 = cv2.imread(img1)
            show_img2 = cv2.imread(img2)

            value = F.pairwise_distance(model(img_t1), model(img_t2)).item()

            fig, axes = plt.subplots(1, 2, figsize=(8, 4))
            fig.suptitle(f"{value:.4f}", fontsize=14, fontweight='bold')

            axes[0].imshow(show_img1)
            axes[0].axis('off')
            axes[0].set_title("Image 1")

            axes[1].imshow(show_img2)
            axes[1].axis('off')
            axes[1].set_title("Image 2")
            plt.tight_layout()
            plt.savefig(f"result{i}.png", dpi=300, bbox_inches='tight')


    test(f"2750/AnnualCrop/AnnualCrop_1.jpg", f"2750/AnnualCrop/AnnualCrop_2.jpg", 0)
    test(f"2750/AnnualCrop/AnnualCrop_1.jpg", f"2750/Forest/Forest_1.jpg",1)
    test(f"2750/AnnualCrop/AnnualCrop_1.jpg",f"2750/AnnualCrop/AnnualCrop_1.jpg",2)

    with torch.no_grad():
        for x_test, y_test in test_loader:
            x_test, y_test = x_test.to(device), y_test.to(device)
            output = model(x_test)
            embeddings = F.normalize(output, p=2, dim=1).cpu().numpy()

            data.append(embeddings)
            labels.append(y_test.cpu().numpy())

        all_embeddings = np.vstack(data)
        all_labels = np.concatenate(labels)

        tsne = TSNE(n_components=2, perplexity=10, random_state=42, init='pca', learning_rate='auto')
        data_2d = tsne.fit_transform(all_embeddings)

        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(data_2d[:, 0], data_2d[:, 1],
                              c=all_labels, cmap='tab20',
                              s=50, alpha=0.7, edgecolors='black', linewidth=0.5)
        plt.colorbar(scatter, label="Person ID", ticks=sorted(np.unique(all_labels)))
        plt.tight_layout()
        plt.savefig("tsne_embeddings.png", dpi=300, bbox_inches='tight')

if __name__ == '__main__':
    main()