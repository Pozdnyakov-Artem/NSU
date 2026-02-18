import cv2
import numpy as np
import torch
import torchvision
from torchvision import transforms
import torch.nn.functional as F

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def open_img(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = torch.from_numpy(img).float() / 255.0
    img = img.permute(2, 0, 1)
    img = img.unsqueeze(0)
    img = normalize(img)

    return img

model = torchvision.models.resnet50(pretrained=True)
model.eval()

layer4_output = None
avgpool_output = None

def get_layer4_output(module,inputs,output):
    global layer4_output
    layer4_output = output

def get_avgpool_output(module,inputs,output):
    global avgpool_output
    avgpool_output = output

hook = model.layer4.register_forward_hook(get_layer4_output)
hook_avg = model.avgpool.register_forward_hook(get_avgpool_output)

img2 = open_img(r"imgs/i.jpg")

img = open_img(r"imgs/abc.png")
orig_img = cv2.imread(r"imgs/abc.png")
orig_img = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
H, W = img.shape[2:]

with torch.no_grad():
    _ = model(img2)
    B, C, h, w = avgpool_output.shape
    avgout = avgpool_output.reshape(B, C, -1).squeeze(0)
    avgout = F.normalize(avgout, p=2, dim=1)
    hook_avg.remove()

    layer4_output = None

    _ = model(img)

    hook.remove()
    B, C, h_feat, w_feat = layer4_output.shape

    output1 = layer4_output.reshape(B, C, -1).permute(0, 2, 1).squeeze(0)
    output1 = F.normalize(output1, p=2, dim=1)

    similarities = torch.matmul(output1, avgout).reshape(h_feat,w_feat).numpy()
    heatmap_norm = (similarities - similarities.min()) / (similarities.max() - similarities.min() + 1e-8)
    similarities = cv2.resize(heatmap_norm*255, (W,H), interpolation=cv2.INTER_CUBIC)

    heatmap = cv2.applyColorMap(similarities.astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    overlay = cv2.addWeighted(orig_img, 0.5, heatmap, 0.5, 0,dtype=cv2.CV_8U)

    cv2.namedWindow('cam', cv2.WINDOW_NORMAL)
    cv2.imshow('cam', cv2.cvtColor(overlay,cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)