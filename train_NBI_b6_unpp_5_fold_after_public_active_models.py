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
from utils.loss_functions import get_loss_function
# from dataloaders import utils
from dataloaders.ISIC2017_dataset import train_dataset,train_transform,test_transform,train_dataset_noread
from networks.utils import create_model
from networks.lib.models import QuickGELU, LayerNorm
from functools import partial
import warnings
from torch.nn.parallel import DataParallel
warnings.filterwarnings('ignore')
import segmentation_models_pytorch as smp
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str,
                    default='//mnt/orton/dataset/1_data_for_label_noise/ISIC2017_noise/train/', help='Name of Experiment')
parser.add_argument('--exp', type=str,
                    default='unpp', help='experiment_name')
parser.add_argument('--model', type=str,
                    default='unet', help='model_name')
parser.add_argument('--num_classes', type=int,  default=2,
                    help='output channel of network')

parser.add_argument('--max_epochs', type=int,
                    default=150, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=6,
                    help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1,
                    help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.0001,
                    help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=2024, help='random seed')
parser.add_argument('--gpu', type=str, default='0', help='0,0.5，0.7')
parser.add_argument('--loss_function', type=str, default='CE', help='CE,GCE,MAE,RCE,SCE,NGCE,NCE_RCE,NGCE_MAE,NGCE_RCE')
parser.add_argument('--model_name', type=str, default='densenet201_unpp', help='CE,GCE,MAE,RCE,SCE,NGCE,NCE_RCE,NGCE_MAE,NGCE_RCE')

args = parser.parse_args()
# os.environ['CUDA_VISIBLE_DEVICES'] = '1,3'
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
def train(args):
    for fold in range(3,5):
        snapshot_path = "./models_5fold/{}".format(args.exp) + '/fold'+str(fold)
        os.makedirs("./models_5fold/{}".format(args.exp), exist_ok=True)
        os.makedirs(snapshot_path,exist_ok=True)

        batch_size = args.batch_size
        path_img = '/mnt/SSD2/NBI息肉分割/all/images_512/'
        path_gt = '//mnt/SSD2/NBI息肉分割/all/masks_512/'
        tr_pdd = pd.read_csv('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/tool_csv/fold'+str(fold)+'_train.csv')
        te_pdd = pd.read_csv('//mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/tool_csv/fold'+str(fold)+'_val.csv')
        names_tr0 = tr_pdd['image_name'].tolist()#[:50]
        active_list = pd.read_csv('/mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models_5fold/active/activate_sample.csv')['image_name'].tolist()
        names_tr = []
        for name in names_tr0:
            names_tr.append(name)
            if name in active_list:
                names_tr.append(name)
        names_test = te_pdd['image_name']#[:50]

        # model = smp.UnetPlusPlus(encoder_name='timm-efficientnet-b6',encoder_weights=None,classes=2).cuda()
        # # model.encoder.load_state_dict(torch.load('./pretrain_weight/tf_efficientnet_b6_aa-80ba17e4.pth'))
        # model.load_state_dict(torch.load('/mnt/orton/codes/race/ISICDM2024/NBI_xi_rou/models/public_popyp_pretrain_b6_unpp/public_popyp_pretrain_b6_unppepoch99.pth'))
        if args.model_name == 'res101_unpp':  # densenet201  timm-resnest101e
            model = smp.UnetPlusPlus(encoder_name='timm-resnest101e',encoder_weights=None,classes=2).cuda()
            model.encoder.load_state_dict(torch.load('/mnt/orton/codes/race/ISICDM2024/shen_xiao_qiu/pretrain_weight/resnest101-22405ba7.pth'))
        if args.model_name == 'densenet201_unpp':  # densenet201  timm-resnest101e
            model = smp.UnetPlusPlus(encoder_name='densenet201',encoder_weights=None,classes=2).cuda()
            model.encoder.load_state_dict(torch.load('/mnt/orton/codes/race/ISICDM2024/shen_xiao_qiu/pretrain_weight/densenet201-5750cbb1e.pth'))
        if args.model_name == 'b6unet':  # densenet201  timm-resnest101e
            model = smp.Unet(encoder_name='timm-efficientnet-b6',encoder_weights=None,classes=2).cuda()
            model.encoder.load_state_dict(torch.load('/mnt/orton/codes/race/ISICDM2024/shen_xiao_qiu/pretrain_weight/tf_efficientnet_b6_aa-80ba17e4.pth'))
        model.cuda()
        # model = DataParallel(model)
        imgs = [cv2.imread(path_img+name.replace('.jpg','.png'))[:,:,::-1]for name in names_tr]
        gts = [cv2.imread(path_gt+name.replace('.jpg','.png'))[:,:,0]/255 for name in names_tr]
        db_train = train_dataset_noread(imgs=imgs, gts = gts, transform=train_transform)

        imgs = [cv2.imread(path_img + name.replace('.jpg','.png'))[:, :, ::-1] for name in names_test]
        gts = [cv2.imread(path_gt + name.replace('.jpg', '.png'))[:, :, 0] / 255 for name in names_test]
        db_test = train_dataset_noread(imgs=imgs, gts = gts,  transform=test_transform)
        def worker_init_fn(worker_id):
            random.seed(args.seed + worker_id)

        trainloader = DataLoader(db_train, batch_size=batch_size, shuffle=True,num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn,drop_last=True)
        testloader = DataLoader(db_test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
        model.train()

        optimizer = optim.AdamW(model.parameters(), lr=args.base_lr)
        CosineLR = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs,eta_min=1e-6)
        # scheduler = StepLR(optimizer, step_size=40, gamma=0.1)
        # optimizer = optim.Adam(model.parameters(), lr=args.base_lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.01)
        # if args.loss_function == 'ce':
        cri = get_loss_function(args)
        from utils import losses
        dice_loss = losses.DiceLoss(2)
        # writer = SummaryWriter(snapshot_path + '/log')
        logging.info("{} iterations per epoch".format(len(trainloader)))

        iter_num = 0
        # max_epoch = max_iterations // len(trainloader) + 1
        max_epoch = args.max_epochs
        iterator = tqdm(range(max_epoch), ncols=70)
        final_dc,final_pre,final_recall,final_f1score,final_pa = [],[],[],[],[]
        best_dice = 0.7
        for iter in iterator:
            model.train()
            for i_batch, sampled_batch in enumerate(trainloader):

                volume_batch, label_batch = sampled_batch
                volume_batch, label_batch = volume_batch.cuda().float(), label_batch.cuda().long()
                P = model(volume_batch)
                loss_P = cri(P, label_batch)
                P_soft = torch.softmax(P,dim=1)
                loss_P_dice = dice_loss(P_soft, label_batch.unsqueeze(1))
                loss = loss_P + loss_P_dice
                # loss.backward()
                # outputs = model(volume_batch)[0]
                # loss = cri(outputs, label_batch)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                iter_num +=1
                # writer.add_scalar('info/total_loss', loss, iter_num)
            # if iter >=max_epoch - 11:
            # if iter >= max_epoch-10:
                # save_mode_path = os.path.join(
                #     snapshot_path, 'epoch' + str(iter) + '.pth')
                # torch.save(model.state_dict(), save_mode_path)
            if iter > 5:
                model.eval()
                with torch.no_grad():
                    test_mdice,test_pre,test_recall ,test_F1score,test_pa= 0,0,0,0,0
                    num = 0
                    for batch_idx, sampled_batch in enumerate(testloader):
                        volume_batch, label_batch = sampled_batch
                        volume_batch, label_batch = volume_batch.cuda().float(), label_batch.cuda().long()
                        out = model(volume_batch)
                        # out_soft =torch.softmax(out,dim=1)[:,1,:,:]
                        predicted = out.argmax(1)

                        test_mdice += Miou.calculate_mdice(predicted, label_batch, 2).item()
                        # test_mdice += dice_coefficient(label_batch, out_soft).item()
                        test_pre += Miou.pre(predicted, label_batch).item()
                        test_recall += Miou.recall(predicted, label_batch).item()
                        test_F1score+=Miou.F1score(predicted, label_batch).item()
                        test_pa += Miou.Pa(predicted, label_batch).item()
                        num +=1

                    average_mdice,ave_pre,ave_recall,ave_f1score,ave_pa = test_mdice/num,test_pre/num,test_recall/num,test_F1score/num,test_pa/num
                    print('fold', fold, 'iter', str(iter), 'average_mdice ', average_mdice)
                    if average_mdice > best_dice:
                        best_dice = average_mdice
                        save_best = os.path.join(snapshot_path,
                                                 'best_model.pth'.format(args.model))
                        torch.save(model.state_dict(), save_best)
                        print('fold',fold,'iter',str(iter),'best_dice ',best_dice)
                        f = open(snapshot_path + '/result.txt', "a")
                        f.write('best dice here'+ '\n')
                        f.close()
                    f = open(snapshot_path +'/result.txt', "a")
                    f.write('epoch' + str(iter) + 'average_mdice' + str(average_mdice) + 'ave_pre' + str(ave_pre)+
                            'ave_recall' + str(ave_recall)  + 'ave_f1score' + str(ave_f1score)+'ave_pa' + str(ave_pa)+'\n')
                    f.close()

                    final_dc.append(average_mdice)
                    final_pre.append(ave_pre)
                    final_recall.append(ave_recall)
                    final_f1score.append(ave_f1score)
                    final_pa.append(ave_pa)
            CosineLR.step()
        f = open(snapshot_path + '/result.txt', "a")
        f.write('\n')
        f.write('final   '+'\n' + 'average_mdice    ' + str(np.mean(np.array(final_dc))) + 'ave_pre   ' + str(np.mean(np.array(final_pre))) +
                'ave_recall ' + str(np.mean(np.array(final_recall))) + 'ave_f1score  ' + str(np.mean(np.array(final_f1score))) + 'ave_pa  ' +
                str(np.mean(np.array(final_pa))) + '\n')
        f.close()
        # logging.info("save model to {}".format(save_mode_path))
        # writer.close()
    return "Training Finished!"


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

    train(args)
