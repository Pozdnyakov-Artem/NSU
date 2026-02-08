import torch
import torchvision
import torchvision.models
import cv2
from matplotlib import pyplot as plt
from torchvision.transforms import transforms

model = torchvision.models.resnet50(pretrained=True)
avgpool_emb = None
features_map = None

def get_embedding(module, inputs, output):
    global avgpool_emb
    avgpool_emb = output

def get_features_map(module, inputs, output):
    global features_map
    features_map = output

model.layer4.register_forward_hook(get_features_map)
model.avgpool.register_forward_hook(get_embedding)
W = model.fc.weight.data
model.eval()

img = cv2.imread(r'2.jpg')

width, height = img.shape[1], img.shape[0]

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = torch.from_numpy(img).float() / 255.0  # [0, 255] → [0, 1]
img = img.permute(2, 0, 1)               # HWC → CHW
img = img.unsqueeze(0)                   # Добавляем батч

# Нормализация под ImageNet
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
img = normalize(img)

model(img)


print(features_map.shape)
print(avgpool_emb.shape)
print(W.shape)

nw = W[250].view(1, 2048, 1, 1)
act = (nw*features_map).sum(dim=1)
cam = torch.relu(act)  # оставляем только положительные активации
cam = cam.squeeze(0)

cam = torch.nn.functional.interpolate(
    cam.unsqueeze(0).unsqueeze(0),  # [1, 1, 7, 7]
    size=(height, width),
    mode='bilinear',
    align_corners=False
).squeeze()

plt.figure(figsize=(12, 4))
plt.imshow(cam.detach().cpu().numpy())

plt.show()