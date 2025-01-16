import numpy as np
import cv2
import os
from PIL import Image

def metrics_EMD(real_path,fake_path):
    # 加载图像
    imgs = []
    for filename in os.listdir(real_path):
        file_path = os.path.join(real_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize((299,299), Image.ANTIALIAS)
            img_resized = np.array(img_resized)
            imgs.append(img_resized)
    imgs = np.array(imgs)

    genimgs = []
    for filename in os.listdir(fake_path):
        file_path = os.path.join(fake_path, filename)
        
        with Image.open(file_path) as img:
            img = img.convert('RGB')
            img_resized = img.resize((299,299), Image.ANTIALIAS)
            img_resized = np.array(img_resized)
            genimgs.append(img_resized)
    genimgs = np.array(genimgs)
    #import pdb;pdb.set_trace()
    hist_real = get_hist(imgs)
    hist_fake = get_hist(genimgs)
    signature_real = hist2signature(hist_real)
    signature_fake = hist2signature(hist_fake)
    retval, lowerBound, flow = cv2.EMD(signature_real, signature_fake, cv2.DIST_L2)
    print(f"EMD:{retval}")
    return retval




def get_hist(img_list):
    hsv_list=[]
    for img_in in img_list:
        img = img_in.copy()
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        hsv_list.append(hsv)
    
    channels = [0, 1, 2]
    histSize = [8, 10, 10]
    ranges = [0, 180, 0, 256, 0, 256]
    hist = cv2.calcHist(hsv_list, channels, None, histSize, ranges)
    hist = cv2.normalize(hist, hist, 1.0, 0.0, cv2.NORM_MINMAX)
    return hist


def hist2signature(hist):
    histSize = hist.shape
    assert len(histSize) == 3
    signature = np.zeros(shape=(histSize[0] * histSize[1] * histSize[2], 4), dtype=np.float32)
    for h in range(histSize[0]):
        for s in range(histSize[1]):
            for v in range(histSize[2]):

                idx = h * 100 + s * 10 + v
                signature[idx][0] = hist[h][s][v]
                signature[idx][1] = h
                signature[idx][2] = s
                signature[idx][3] = v
    return signature
