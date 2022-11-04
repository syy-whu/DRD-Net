import cv2
import os
import argparse
import glob
import numpy as np
import torch
from torch.autograd import Variable
from utils import *
from networks import *
from skimage.measure import compare_psnr, compare_ssim
import time
from logs.pixel_metric import metric

parser = argparse.ArgumentParser(description="DRDNet_Test")
parser.add_argument("--logdir", type=str, default="logs/Rain200L/", help='path to model and log files')
parser.add_argument("--data_path", type=str, default="logs/test/Rain200L/rain/X2", help='path to training data')
parser.add_argument("--gt_path", type=str, default="logs/test/Rain200L/norain", help='path to gt data for metric')
parser.add_argument("--save_path", type=str, default="logs/test/Rain200L", help='path to save results')
parser.add_argument("--use_GPU", type=bool, default=True, help='use GPU or not')
parser.add_argument("--gpu_id", type=str, default="0", help='GPU id')
parser.add_argument("--recurrent_iter", type=int, default=6, help='number of recursive stages')
opt = parser.parse_args()

if opt.use_GPU:
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpu_id


def main():

    os.makedirs(opt.save_path, exist_ok=True)

    # Build model
    print('Loading model ...\n')
    model = DRDNetwork(is_train=False)
    print_network(model)
    if opt.use_GPU:
        model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.logdir, 'net_epoch100.pth')))
    model.eval()

    time_test = 0
    count = 0
    for img_name in os.listdir(opt.data_path):
        if is_image(img_name):
            img_path = os.path.join(opt.data_path, img_name)

            # input image
            y = cv2.imread(img_path)
            b, g, r = cv2.split(y)
            y = cv2.merge([r, g, b])
            #y = cv2.resize(y, (int(500), int(500)), interpolation=cv2.INTER_CUBIC)

            y = normalize(np.float32(y))
            y = np.expand_dims(y.transpose(2, 0, 1), 0)
            y = Variable(torch.Tensor(y))

            if opt.use_GPU:
                y = y.cuda()

            with torch.no_grad(): #
                if opt.use_GPU:
                    torch.cuda.synchronize()
                start_time = time.time()

                out1, out, back = model(y)
                out = torch.clamp(out, 0., 1.)
                back = torch.clamp(back, 0., 1.)

                if opt.use_GPU:
                    torch.cuda.synchronize()
                end_time = time.time()
                dur_time = end_time - start_time
                time_test += dur_time

                print(img_name, ': ', dur_time)

            if opt.use_GPU:
                save_out = np.uint8(255 * out.data.cpu().numpy().squeeze())   #back to cpu
                back_out = np.uint8(255 * back.data.cpu().numpy().squeeze())
            else:
                save_out = np.uint8(255 * out.data.numpy().squeeze())
                back_out = np.uint8(255 * back.data.numpy().squeeze())


            save_out = save_out.transpose(1, 2, 0)
            back_out = back_out.transpose(1, 2, 0)

            b, g, r = cv2.split(save_out)
            save_out = cv2.merge([r, g, b])

            b, g, r = cv2.split(back_out)
            back_out = cv2.merge([r, g, b])

            out_save_path=opt.save_path + '/out_result'
            back_save_path=opt.save_path + '/back_result'
            if not os.path.exists(out_save_path):
                os.makedirs(out_save_path)
            if not os.path.exists(back_save_path):
                os.makedirs(back_save_path)
            cv2.imwrite(os.path.join(out_save_path, img_name), save_out)
            cv2.imwrite(os.path.join(back_save_path, img_name), back_out)

            count += 1

    print('Avg. time:', time_test/count)
    ### metric
    metric(out_save_path,opt.gt_path)



if __name__ == "__main__":
    main()

