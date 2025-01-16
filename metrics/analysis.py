from FID import metrics_FID
from IS import metrics_IS
from ICLPIPS import metrics_ICLPIPS
from KID import metrics_KID
from EMD import metrics_EMD
from KL_DIS import metrics_KL
from JS_DIS import metrics_JS
import numpy as np
import pandas as pd
import os

data_root = "/data4/jiangtianjia/datasets/mvtec"
fake_data_root = "/data4/jiangtianjia/datasets/refdata/anogen_gen"
save_dir="./anogen100_results.csv"

for cls in os.listdir(data_root):
    classpath = os.path.join(data_root,cls)
    if(os.path.isfile(classpath)):
        continue
    for anomaly in os.listdir(os.path.join(classpath,'ground_truth')):
        anomaly_path = os.path.join(data_root,cls,'test',anomaly)
        fake_path=os.path.join(fake_data_root,cls,anomaly,'image')
        print(f'\n\nReal image from "{anomaly_path}" and Fake image from "{fake_path}" \n\n')

        #FID计算两个特征分布的差异，越小越好
        #dims=[64, 192, 768, 2048]
        fid64 = metrics_FID(anomaly_path,fake_path,device="cuda",dims=64)

        #感知质量：LPIPS评估了生成图像的感知质量，即图像与真实图像在感知上的相似度。
        # Intra-cluster pairwise LPIPS评估类内两个图像的差异，越大越显示生成差异性大。
        #越大越好
        iclpips = metrics_ICLPIPS(anomaly_path,fake_path)

        # 高Inception Score意味着生成的图像既具有多样性（不同图像属于不同类别），
        # 又具有清晰度（单张图像的类别概率分布集中）。数值越大表示生成图像的质量越好
        inception_score=metrics_IS(fake_path)

        #KID的数值越低，表示生成图像与真实图像在特征空间中的分布越接近，即GAN的性能越好。
        kid=metrics_KID(anomaly_path,fake_path,batch_size=4)


        #Earth Move Distance 越小越像
        emd=metrics_EMD(anomaly_path,fake_path)

        #计算KL 越小越像 非对称
        kl=metrics_KL(anomaly_path,fake_path)

        #计算JS 越小越像 对称
        js=metrics_JS(anomaly_path,fake_path)

        # 创建DataFrame
        df = pd.DataFrame([])
        path = save_dir
        if(os.path.exists(path)):
            df = pd.read_csv(path)
        new_row_df = pd.DataFrame([{'anomaly': f'{cls}_{anomaly}', "IS↑":f"{inception_score:4f}",'FID64↓': f"{fid64:4f}", 'ICLPIPS↓': f"{iclpips:4f}" ,"KID↓":f"{kid:4f}" ,"EMD↓":f"{emd:4f}", "KL↓":f"{kl:4f}", "JS↓":f"{js:4f}"}])
        df = pd.concat([df, new_row_df], ignore_index=True)
        # 保存为CSV文件
        csv_file_path = path  # 替换为您的文件路径
        df.to_csv(csv_file_path, index=False, encoding='utf-8')

        

