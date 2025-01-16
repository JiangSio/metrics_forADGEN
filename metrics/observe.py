from FID import metrics_FID
from IS import metrics_IS
from ICLPIPS import metrics_ICLPIPS
from KID import metrics_KID
from EMD import metrics_EMD
from KL_DIS import metrics_KL
from JS_DIS import metrics_JS
path_real_images = '/data4/jiangtianjia/datasets/mvtec/screw/test/scratch_head'  # 真实图像的路径
path_fake_images = '/data4/jiangtianjia/datasets/mvtec/bottle/test/broken_small'  # 生成图像的路径

#FID计算两个特征分布的差异，越小越好
#dims=[64, 192, 768, 2048]
metrics_FID(path_real_images,path_fake_images,device="cuda",dims=64)

#感知质量：LPIPS评估了生成图像的感知质量，即图像与真实图像在感知上的相似度。
# Intra-cluster pairwise LPIPS评估类内两个图像的差异，越大越显示生成差异性大。
#越大越好
metrics_ICLPIPS(path_real_images,path_fake_images)

# 高Inception Score意味着生成的图像既具有多样性（不同图像属于不同类别），
# 又具有清晰度（单张图像的类别概率分布集中）。数值越大表示生成图像的质量越好
metrics_IS(path_fake_images)

#KID的数值越低，表示生成图像与真实图像在特征空间中的分布越接近，即GAN的性能越好。
metrics_KID(path_real_images,path_fake_images,batch_size=4)


#Earth Move Distance 越小越像
metrics_EMD(path_real_images,path_fake_images)

#计算KL 越小越像 非对称
metrics_KL(path_real_images,path_fake_images)

#计算JS 越小越像 对称
metrics_JS(path_real_images,path_fake_images)

