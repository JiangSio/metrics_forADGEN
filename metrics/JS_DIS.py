import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import tensor
import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from torch.nn import functional as F
import torch.utils.data
from torchvision.models.inception import inception_v3
from torchmetrics.regression import KLDivergence
import random

preprocess = transforms.Compose([
    transforms.Resize((299,299)),  # Inception v3期望的输入尺寸是299x299
    transforms.CenterCrop((299,299)),
    transforms.ToTensor(),  # 将图片转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

inception_model = inception_v3(pretrained=True, transform_input=False).cuda()
inception_model.eval()

def metrics_JS(real_path,fake_path):
    imgs = []
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = preprocess(img)
            imgs.append(img)

    genimgs = []
    for filename in os.listdir(fake_path):
        file_path = os.path.join(fake_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img = preprocess(img)
            genimgs.append(img)

    test_len = min(len(imgs),len(genimgs))
    random.shuffle(imgs)
    imgs = imgs[:test_len]
    random.shuffle(genimgs)
    genimgs=genimgs[:test_len]

    imgs = torch.stack(imgs).cuda()
    print(imgs.shape)

    genimgs = torch.stack(genimgs).cuda()
    print(genimgs.shape)

    with torch.no_grad():
        imgs_embedding = inception_model(imgs)
        genimgs_embedding = inception_model(genimgs)
    imgs_embedding = torch.nn.functional.softmax(imgs_embedding, dim=1).cpu().numpy()
    genimgs_embedding = torch.nn.functional.softmax(genimgs_embedding, dim=1).cpu().numpy()

    kl_divergence = KLDivergence()
    imgs_embedding=imgs_embedding.mean(axis=0)
    imgs_embedding=imgs_embedding[np.newaxis, :]
    genimgs_embedding=genimgs_embedding.mean(axis=0)
    genimgs_embedding=genimgs_embedding[np.newaxis, :]
    mean_embedding = (imgs_embedding+genimgs_embedding)/2
    res1 = kl_divergence(tensor(imgs_embedding),tensor(mean_embedding))
    res2 = kl_divergence(tensor(genimgs_embedding),tensor(mean_embedding))
    res = 0.5*res1 + 0.5*res2
    print(f"JS_Divergence:{res}")
    return res.item()