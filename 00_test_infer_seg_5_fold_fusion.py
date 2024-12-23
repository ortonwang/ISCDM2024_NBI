import argparse
import logging
import os
import random

import pandas as pd

from utils import Miou
import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
from utils.demo_dice import dice_coefficient
import torch.nn.functional as F
import torch.optim as optim
# from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='//mnt/orton/dataset/1_data_for_label_noise/ISIC2017_noise/train/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='debug', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

parser.add_argument('--max_epochs', type=int,
                    default=200, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=8,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.001,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--gpu', type=str, default='3', help='0,0.5，0.7')
parser.add_argument('--loss_function', type=str, default='CE', help='CE,GCE,MAE,RCE,SCE,NGCE,NCE_RCE,NGCE_MAE,NGCE_RCE')

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,2,3'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import segmentation_models_pytorch as smp
import torchvision
rot = torchvision.transforms.functional.rotate


def sum_8(model, tensor):
    batch = torch.cat([tensor, tensor.flip([-1]), tensor.flip([-2]), tensor.flip([-1, -2]), rot(tensor, 90).flip([-1]), rot(tensor, 90).flip([-2]), rot(tensor, 90).flip([-1, -2]),
                       rot(tensor, 90)], 0)
    pred = model(batch).detach()
    return pred[:1] + pred[1:2].flip([-1]) + pred[2:3].flip([-2]) + pred[3:4].flip([-1, -2]) + rot(pred[4:5].flip([-1]), -90) + rot(pred[5:6].flip([-2]), -90) + rot(
        pred[6:7].flip([-1, -2]), -90) + rot(pred[7:], -90)
from albumentations.pytorch import ToTensorV2
import albumentations as A
test_transform = A.Compose([
     ToTensorV2()], p=1.)
def train(args, snapshot_path):

    model0 = smp.Unet(encoder_name='mit_b5', encoder_weights=None, classes=2).cuda()
    model0.load_state_dict(torch.load('///mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/mit_b5unet_5_fold_pre_public_and_active/fold0/best_model.pth'))

    model01 = smp.Unet(encoder_name='mit_b5', encoder_weights=None, classes=2).cuda()
    model01.load_state_dict(torch.load('///mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/mit_b5unet_5_fold_pre_public_and_active/fold1/best_model.pth'))

    model02 = smp.Unet(encoder_name='mit_b5', encoder_weights=None, classes=2).cuda()
    model02.load_state_dict(torch.load('///mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/mit_b5unet_5_fold_pre_public_and_active/fold2/best_model.pth'))

    model03 = smp.Unet(encoder_name='mit_b5', encoder_weights=None, classes=2).cuda()
    model03.load_state_dict(torch.load('///mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/mit_b5unet_5_fold_pre_public_and_active/fold3/best_model.pth'))

    model04 = smp.Unet(encoder_name='mit_b5', encoder_weights=None, classes=2).cuda()
    model04.load_state_dict(torch.load('///mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/mit_b5unet_5_fold_pre_public_and_active/fold4/best_model.pth'))

    model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b6', encoder_weights=None, classes=2).cuda()
    model.load_state_dict(torch.load('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/b6_unpp_5_fold_pre_public_and_active_lr3e4/fold0/best_model.pth'))
    model11 = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b6', encoder_weights=None, classes=2).cuda()
    model11.load_state_dict(torch.load('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/b6_unpp_5_fold_pre_public_and_active_lr3e4/fold1/best_model.pth'))
    model12 = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b6', encoder_weights=None, classes=2).cuda()
    model12.load_state_dict(torch.load('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/b6_unpp_5_fold_pre_public_and_active_lr3e4/fold2/best_model.pth'))
    model13 = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b6', encoder_weights=None, classes=2).cuda()
    model13.load_state_dict(torch.load('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/b6_unpp_5_fold_pre_public_and_active_lr3e4/fold3/best_model.pth'))
    model14 = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b6', encoder_weights=None, classes=2).cuda()
    model14.load_state_dict(torch.load('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/b6_unpp_5_fold_pre_public_and_active_lr3e4/fold4/best_model.pth'))

    models = []
    models.append(model0)
    models.append(model01)
    models.append(model02)
    models.append(model03)
    models.append(model04)
    models.append(model)
    models.append(model11)
    models.append(model12)
    models.append(model13)
    models.append(model14)

    names_test = os.listdir('//mnt/SSD2/NBI息肉分割/测试的提交/挑战赛项目1-测试代码/test/images/')#[:5]
    path_img_test = '/mnt/SSD2/NBI息肉分割/测试的提交/挑战赛项目1-测试代码/test/images/'

    result_dir = '//mnt/SSD2/NBI息肉分割/测试的提交/挑战赛项目1-测试代码/test/pre1/'

    with torch.no_grad():
        for name in tqdm(names_test):
            img = cv2.imread(path_img_test + name)[:, :, ::-1]
            shape = img.shape[:2]
            img_res = cv2.resize(img,[512,512])
            img_res_tensor = test_transform(image=img_res.copy())['image']
            img_res_tensor = torch.unsqueeze(img_res_tensor,0).cuda().float()
            results = torch.zeros(1, 2, 512, 512, device='cuda')
            for model in models:
                model.to('cuda').eval()
                results += sum_8(model, img_res_tensor)

            predicted0 = results.argmax(1)
            out = predicted0[0].to('cpu').numpy()
            cv2.imwrite(result_dir + '/' + name.replace('.jpg', '.png'),cv2.resize((out*255), shape[::-1], interpolation=cv2.INTER_NEAREST))
            # print('fds')
            # cv2.imwrite(result_dir + '/' + image_name.replace('.jpg', '.png'), out * 255)





if __name__ == "__main__":
    if not args.deterministic:
        cudnn.benchmark = True
        cudnn.deterministic = False
    else:
        cudnn.benchmark = False
        cudnn.deterministic = True

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    snapshot_path = "./models/{}".format(args.exp)
    os.makedirs(snapshot_path,exist_ok=True)
    train(args, snapshot_path)
