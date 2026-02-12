import numpy as np
import torch
import torch.nn.functional as F
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

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
frame = cv2.imread(r'2.jpg')
# cap = cv2.VideoCapture(0)

while True:

    # ret, frame = cap.read()
    img = frame
    width, height = img.shape[1], img.shape[0]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)

    img = normalize(img)

    output = model(img)
    _, predicted_idx = torch.max(output, 1)
    print(predicted_idx)


    print(features_map.shape)   #1 2048 15 20
    print(avgpool_emb.shape)    #1 2048 1 1
    # print(W.shape, W[250].shape)
    nw = W[predicted_idx].view(1, 2048, 1, 1) #250
    # print(nw.shape)

    act = (nw*features_map).sum(dim=1)
    # print(min(act))
    # act = F.conv2d(features_map, W.unsqueeze(-1).unsqueeze(-1))
    # cam = torch.relu(act)
    cam = act.squeeze(0)
    cam = cam.squeeze(0)
    # print(cam.shape)
    cam = torch.nn.functional.interpolate(
        cam.unsqueeze(0).unsqueeze(0),
        size=(height, width),
        mode='bilinear',
        align_corners=False
    ).squeeze()

    cam = cam - cam.min()
    cam = cam / (cam.max())
    cam = cam * 255
    cam = cam.detach().numpy().astype(np.uint8)
    heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(frame, 0.5, heatmap, 0.5, 0)
    cv2.imshow('cam', overlay)
    cv2.waitKey(1)
    # plt.figure(figsize=(12, 4))
    # plt.imshow(cam.detach().numpy())

    # plt.show()