import torch
import torchvision
import torchvision.transforms as transforms
from pytorch_fid import fid_score
from torchmetrics.image.fid import FrechetInceptionDistance
from PIL import Image
import numpy as np
import os
import random

def metrics_FID(real_path,fake_path,device="cuda",dims=64):
    # 计算FID
    #fid_value = fid_score.calculate_fid_given_paths(paths=[real_path, fake_path], batch_size=batch_size, device=device, dims=dims)

    # 打印FID值
    #print(f"FID score: {fid_value}")

    imgs = []
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize((299,299), Image.ANTIALIAS)
            img_resized = np.array(img_resized)
            imgs.append(img_resized)

    genimgs = []
    for filename in os.listdir(fake_path):
        file_path = os.path.join(fake_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize((299,299), Image.ANTIALIAS)
            img_resized = np.array(img_resized)
            genimgs.append(img_resized)

    test_len = min(len(imgs),len(genimgs))
    random.shuffle(imgs)
    imgs = imgs[:test_len]
    random.shuffle(genimgs)
    genimgs=genimgs[:test_len]
    #import pdb;pdb.set_trace()
    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs).permute(0,3,1,2).to(dtype=torch.uint8)
    print(imgs.shape)
    genimgs = np.array(genimgs)
    genimgs = torch.from_numpy(genimgs).permute(0,3,1,2).to(dtype=torch.uint8)
    print(genimgs.shape)

    fid = FrechetInceptionDistance(feature=dims)
    fid.update(imgs, real=True)
    fid.update(genimgs, real=False)
    res=fid.compute()
    print(f"FID score2: {res}")
    return res.item()


