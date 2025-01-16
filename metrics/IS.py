from torchmetrics.image.inception import InceptionScore
import torch
from PIL import Image
import os
import numpy as np
import random
MAX_SIZE = 500

def metrics_IS(fake_path):
    imgs = []
    for filename in os.listdir(fake_path):
        file_path = os.path.join(fake_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize((299,299), Image.ANTIALIAS)
            img_resized = np.array(img_resized)
            imgs.append(img_resized)

    test_len = min(MAX_SIZE,len(imgs))
    random.shuffle(imgs)
    imgs = imgs[:test_len]

    imgs = np.array(imgs)
    imgs = torch.from_numpy(imgs).permute(0,3,1,2).to(dtype=torch.uint8)
    print(imgs.shape)

    inception = InceptionScore()
    inception.update(imgs)
    res = inception.compute()

    print(f"Inception Score:(mean/std): {res}")
    return res[0].item()

 
 