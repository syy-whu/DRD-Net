import cv2
import math
import numpy as np
from skimage.measure import compare_psnr, compare_ssim
import os

def psnr(img1, img2):
    mse = np.mean((img1 / 255. - img2 / 255.) ** 2)  # 均方差
    if mse < 1.0e-10:  # 几乎无差异返回100
        return 100
    PIXEL_MAX = 1  # 像素最大值
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

def metric(rain_path,gt_path):
    images_name = os.listdir(rain_path)
    psnr_sum = 0
    ssim_sum = 0
    i = 0
    # 遍历所有文件名
    for eachname in images_name:
        # 按照规则将内容写入txt文件中

        strlist = eachname.split('x')

        trainname = rain_path + "/" + eachname
        gtname = gt_path + "/" + strlist[0] + '.png'

        img1 = cv2.imread(trainname)
        img2 = cv2.imread(gtname)
        psnrnum = compare_psnr(img1, img2, data_range=255)
        ssimnum = compare_ssim(img1, img2, data_range=255, multichannel=True)
        psnr_sum = psnr_sum + psnrnum
        ssim_sum = ssim_sum + ssimnum
        i = i + 1
        print(i)
        print(trainname + ":", psnrnum)
        print(trainname + ":", ssimnum)
    finalpsnr = psnr_sum / i
    fianlssim = ssim_sum / i
    print("Average PSNR: %f "%finalpsnr)
    print("Average SSIM: %f "%fianlssim)


if __name__ == "__main__":
    MOR_path = "test/Rain800/out_result"
    gt_path = "test/Rain800/norain"
    images_name = os.listdir(MOR_path)
    psnr_sum=0
    ssim_sum = 0
    i=0
    # 遍历所有文件名
    for eachname in images_name:
        # 按照规则将内容写入txt文件中

        strlist = eachname.split('x')

        trainname = MOR_path + "/" + eachname
        # gtname = gt_path + "/" + strlist[0] + '.png'
        gtname = gt_path + "/" + eachname

        img1 = cv2.imread(trainname)
        img2 = cv2.imread(gtname)
        psnrnum=compare_psnr(img1, img2, data_range=255)
        ssimnum=compare_ssim(img1, img2, data_range=255, multichannel=True)
        psnr_sum=psnr_sum+psnrnum
        ssim_sum=ssim_sum+ssimnum
        i=i+1
        print(i)
        print(trainname+":", psnrnum)
        print(trainname+":", ssimnum)
    finalpsnr=psnr_sum/i
    fianlssim=ssim_sum/i
    print(finalpsnr)
    print(fianlssim)