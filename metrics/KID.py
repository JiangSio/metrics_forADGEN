import torch
from torch import randint
from torchmetrics.image.kid import KernelInceptionDistance
import os
from PIL import Image
import numpy as np
import random

def metrics_KID(real_path,fake_path,batch_size=50):

    kid = KernelInceptionDistance(subset_size=batch_size)
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

    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs).permute(0,3,1,2).to(dtype=torch.uint8)
    print(imgs.shape)
    genimgs = np.array(genimgs)
    genimgs = torch.from_numpy(genimgs).permute(0,3,1,2).to(dtype=torch.uint8)
    print(genimgs.shape)

    kid.update(imgs, real=True)
    kid.update(genimgs, real=False)
    res = kid.compute()
    print(f"KID:(mean/std) {res}")
    return res[0].item()