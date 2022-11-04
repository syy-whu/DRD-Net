# DRD-Net(pytorch)-

This repository contains the Pytorch code for the paper Detail-recovery Image Deraining via Context Aggregation Networks (CVPR 2020) \[[arXiv(https://arxiv.org/pdf/1908.10267.pdf)\]. The code of Semi-DRDNet will be coming soon!

## Prerequisites
- Python 3.6, PyTorch >= 0.4.0 
- Requirements: opencv-python, tensorboardX
- Platforms: Ubuntu 16.04, cuda-8.0 & cuDNN v-5.1 (higher versions also work well)

## Dataset preparation
Please download the official [Rain200](https://www.icst.pku.edu.cn/struct/Projects/joint_rain_removal.html) dataset and organize the downloaded files as follows: 
```
DRD-Net
├── datasets
│   ├── Rain200H
│   │   ├── rain
│   │   ├── norain
│   ├── Rain200L
│   │   ├── rain
│   │   ├── norain
│   ├── Rain800
│   │   ├── rain
│   │   ├── norain
```
## Getting Started
### Training
Run DRD-Net on Rain200L:
```shell
CUDA_VISIBLE_DEVICES=0 python train.py --save_path "logs/Rain200L" --data_path "datasets/Rain200L"
```
Others training settings are similar.
### Testing
```shell
CUDA_VISIBLE_DEVICES=0 python test.py --save_path "logs/Rain200L" --data_path "datasets/Rain200L"
```
## Acknowledgement
The code is based on [PReNet](https://github.com/csdwren/PReNet). 

## Citation
If you find this work useful in your research, please consider cite:

```
@inproceedings{deng2020detail,
  title={Detail-recovery image deraining via context aggregation networks},
  author={Deng, Sen and Wei, Mingqiang and Wang, Jun and Feng, Yidan and Liang, Luming and Xie, Haoran and Wang, Fu Lee and Wang, Meng},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={14560--14569},
  year={2020}
}
```
```
@article{shen2022semi,
  title={Semi-DRDNet Semi-supervised Detail-recovery Image Deraining Network via Unpaired Contrastive Learning},
  author={Shen, Yiyang and Deng, Sen and Yang, Wenhan and Wei, Mingqiang and Xie, Haoran and Zhang, XiaoPing and Qin, Jing and Wang, Meng},
  journal={arXiv preprint arXiv:2204.02772},
  year={2022}
}
```
```
 @inproceedings{ren2019progressive,
   title={Progressive Image Deraining Networks: A Better and Simpler Baseline},
   author={Ren, Dongwei and Zuo, Wangmeng and Hu, Qinghua and Zhu, Pengfei and Meng, Deyu},
   booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
   year={2019},
 }
```
