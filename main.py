import os
import math
import torch
import time
import random
import datetime
import argparse
import numpy as np
import pandas as pd
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from decimal import Decimal, ROUND_HALF_UP
from scheduler import GradualWarmupScheduler
from data_provider_men import TB_Dataset as TB_Dataset_men
from data_provider_mrnet import TB_Dataset as TB_Dataset_mrnet
from data_provider_brats2021 import TB_Dataset as TB_Dataset_brats
from sklearn.metrics import roc_curve, accuracy_score, f1_score, roc_auc_score
from sklearn.metrics import auc as AUC
from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, average_precision_score
from torch.utils.data.distributed import DistributedSampler
from models.CFDL import CFDL



parser = argparse.ArgumentParser()
parser.add_argument('--seed', type=int, default=666)
parser.add_argument('--method_name', type=str, default='CFDL')# CFDL
parser.add_argument('--dataset_name', type=str, default='men') # brats, men, mrnet-meniscus
parser.add_argument('--image_type_brats', type=str, default='bbox') # bbox, whole
parser.add_argument('--run_type', type=str, default='train') # train, test
parser.add_argument('--batch_size_men', type=int, default=32)
parser.add_argument('--batch_size_mrnet', type=int, default=32)
parser.add_argument('--batch_size_brats', type=int, default=32)
parser.add_argument('--feature_dim', type=int, default=32)
parser.add_argument('--multi_gpus', type=int, default=0)
parser.add_argument('--gpu', type=int, default=2)
parser.add_argument('--ifbalanceloader', type=int, default=0)
parser.add_argument('--num_workers', type=int, default=12)
parser.add_argument('--pretrained', type=int, default=0)
parser.add_argument('--finetuning', type=int, default=0)
parser.add_argument('--ft_rate', type=float, default=0.01)
parser.add_argument('--model_depth', type=int, default=18)
parser.add_argument('--fold_train', type=int, default=123)
parser.add_argument('--train_lr_men', type=float, default=0.0005)
parser.add_argument('--train_lr_brats', type=float, default=0.0002)
parser.add_argument('--train_lr_mrnet', type=float, default=0.0008)
parser.add_argument('--lr_gamma', type=float, default=0.8)
parser.add_argument('--cls_drop', type=float, default=0.5)
parser.add_argument('--encoder_drop', type=float, default=0.5)
parser.add_argument('--map_drop', type=float, default=0.5)
parser.add_argument('--weight_decay_value', type=float, default=0.0001)
parser.add_argument('--loss_type', type=str, default='ce') # ce, focal
parser.add_argument('--focal_alpha', type=float, default=0.25)
parser.add_argument('--focal_gamma', type=float, default=2)
parser.add_argument('--mixup', type=bool, default=False)
parser.add_argument('--encoder_share', type=bool, default=False)
parser.add_argument('--testTraindata', type=int, default=0) # test_traindata
parser.add_argument('--train_epochs', type=int, default=100)
parser.add_argument('--train_epochs_mrnet', type=int, default=50)
parser.add_argument('--train_epochs_brats', type=int, default=50)
parser.add_argument('--train_epochs_BRCA', type=int, default=1000)
parser.add_argument('--train_epochs_abide', type=int, default=800)
parser.add_argument('--lambda_epochs', type=int, default=30)
parser.add_argument('--dif_weight_type', type=str, default=None)
parser.add_argument('--en_disnum', type=int, default=5)
parser.add_argument('--fusion_type', type=str, default='moe') # concat, moe, sa, fc_sa, conv_sa, cma, tsa(self attention with transformer), DCAF
parser.add_argument('--cma_type', type=str, default='cascade_comp_inf') # ab(ablation study,三个k concat然后和qv做attention), multi_q+map, comp_inf, cascade_comp_inf, cascade_mutual_spe
parser.add_argument('--attention_type', type=str, default='sigmoid') # sigmoid
parser.add_argument('--server', type=str, default='149') # self, wenwu, cuhk, web, label, 149
parser.add_argument('--test_multi', type=int, default=0) # 测试时导入的模型是否是多卡跑的
parser.add_argument('--ifl1', type=int, default=0)
parser.add_argument('--ifdisen', type=int, default=1)
parser.add_argument('--if_offline_data_aug', type=int, default=1)
parser.add_argument('--ifshare_linear_share', type=int, default=1)
parser.add_argument('--ifdis_sup', type=int, default=1)
parser.add_argument('--if_allpair', type=int, default=1)
parser.add_argument('--ifconloss', type=int, default=1)
parser.add_argument('--ifsa_qkv', type=int, default=0)
parser.add_argument('--ifshare_sa', type=int, default=0)
parser.add_argument('--ifaux_spec', type=int, default=0)
parser.add_argument('--ifaux_share', type=int, default=0)
parser.add_argument('--ifrecon', type=int, default=0)
parser.add_argument('--ifcross_recon', type=int, default=0)
parser.add_argument('--ifdiffloss', type=int, default=0)
parser.add_argument('--ifalign_space', type=int, default=0)
parser.add_argument('--ifse_agg', type=int, default=0)
parser.add_argument('--cascade_order', type=str, default='order') # order, reverse
parser.add_argument('--top_k', type=int, default=2) # ADCCA top_k
parser.add_argument('--display_num_try', type=int, default=0)
parser.add_argument('--train_with_noise', type=int, default=0)
parser.add_argument('--test_type', type=str, default='ori')

parser.add_argument('--w_main_cls', type=float, default=1)
parser.add_argument('--w_gating', type=float, default=1)
parser.add_argument('--w_aux', type=float, default=1)
parser.add_argument('--w_spec', type=float, default=0)
parser.add_argument('--w_share', type=float, default=0)
parser.add_argument('--w_recon', type=float, default=1)
parser.add_argument('--w_cross_recon', type=float, default=0)
parser.add_argument('--w_diff', type=float, default=1)
parser.add_argument('--w_con', type=float, default=1)
parser.add_argument('--w_whiten', type=float, default=0)
parser.add_argument('--w_sim', type=float, default=1)

parser.add_argument('--out_channels', type=float, default=16)
parser.add_argument('--threshold', type=float, default=0.5)
parser.add_argument('--tau', type=float, default=0.07)
parser.add_argument('--lambda_maml', type=float, default=0)
parser.add_argument('--sim_loss_type', type=str, default='l2norm') # cosine,MI,l2norm
parser.add_argument('--diff_loss_type', type=str, default='cosine') # orthogonality,l2norm, cosine
parser.add_argument('--share_fusion_type', type=str, default='mean') # mean,concat
parser.add_argument('--share_sup_fusion_type', type=str, default='whiten') # concat, at(autoencoder), whiten
parser.add_argument('--share_share_sup_fusion_type', type=str, default='concat')
parser.add_argument('--save_epoch_start', type=int, default=5)
parser.add_argument('--early', type=int, default=1000)
parser.add_argument('--step_size', type=int, default=100)
parser.add_argument('--ifwarmup', type=bool, default=True)

parser.add_argument('--gate_type', type=str, default='train_gen')
parser.add_argument('--dfs_fusion_type', type=str, default='concat')
parser.add_argument('--mda_balance', type=int, default=0)

parser.add_argument('--vl_hidden', type=int, default=64)
parser.add_argument('--n_head', type=int, default=2)
parser.add_argument('--vl_dropout', type=float, default=0.5)
parser.add_argument('--vl_nlayer', type=int, default=1)
parser.add_argument('--encoder_type', type=str, default='transformer') # transformer, mlp

parser.add_argument('--cma_vl_hidden', type=int, default=64)
parser.add_argument('--cma_n_head', type=int, default=1)
parser.add_argument('--cma_vl_dropout', type=float, default=0.5)
parser.add_argument('--uncertainty', type=int, default=0)
parser.add_argument('--annealing_step', type=int, default=30)
parser.add_argument('--online_aug_type', type=str, default='random_crop')
# moe fusion
parser.add_argument("--moe_fusion_type", type=str, default='concat') # concat, sum
parser.add_argument("--gating_type", type=str, default='importance') # concat, importance, correlation
parser.add_argument("--gating_concat_mlp_layer", type=str, default='multi') # multi, single
parser.add_argument("--ifmoe_aux", type=int, default=0)
parser.add_argument("--w_moe_aux", type=float, default=0.5)
parser.add_argument("--if_ccl", type=int, default=0)
parser.add_argument("--w_ccl", type=float, default=1)

args = parser.parse_args()

ifbalanceloader = args.ifbalanceloader
if ifbalanceloader:
    args.if_offline_data_aug=0


if args.multi_gpus:
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'
    local_rank = int(os.environ["LOCAL_RANK"])

    if local_rank != -1:
        torch.cuda.set_device(local_rank)
        device=torch.device("cuda", local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)


if 'mrnet' in args.dataset_name:
    train_epochs = args.train_epochs_mrnet
elif 'brats' in args.dataset_name:
    train_epochs = args.train_epochs_brats
else:
    train_epochs = args.train_epochs

args.vis_epoch = train_epochs

if args.dataset_name == 'men' or 'mrnet' in args.dataset_name:
    if args.dataset_name == 'men':
        num_classes = 3
        train_lr = args.train_lr_men
        batch_size = args.batch_size_men
    elif 'mrnet' in args.dataset_name:
        num_classes = 2
        train_lr = args.train_lr_mrnet
        batch_size = args.batch_size_mrnet
    modal_num = 3
    image_size = [24, 128, 128]
    window_size = [3, 4, 4]
    if args.if_offline_data_aug:
        display_num = 50
    else:
        display_num = 20
elif args.dataset_name == 'brats':
    num_classes = 2
    if args.multi_gpus:
        batch_size = 11
    else:
        batch_size = args.batch_size_brats
    modal_num = 4
    image_size = [16, 128, 128]
    window_size = [2, 4, 4]
    train_lr = args.train_lr_brats
    if args.if_offline_data_aug:
        display_num = 50
    else:
        display_num = 8

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
torch.cuda.manual_seed_all(args.seed)  # if you are using multi-GPU.
np.random.seed(args.seed)  # Numpy module.
random.seed(args.seed)  # Python random module.
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = True


cuda = True if torch.cuda.is_available() else False


def one_hot(x, class_count):
    # return np.eye(class_count)[x,:]
    return (torch.eye(class_count)[x,:]).cuda()


def one_hot_forauc(x, class_count):
    return np.eye(class_count)[x,:]


def val(model,test_data=None,labels=None):
    model.eval()

    if args.dataset_name == 'men':
        test_set = TB_Dataset_men('test', use_path, False, ifbalanceloader, 'ori', image_type, num_classes, ifonline_aug=False, val=True, args=args)
    elif 'brats' in args.dataset_name:
        test_set = TB_Dataset_brats('test', use_path, False, ifbalanceloader, 'ori', False, num_classes, ifonline_aug=False, val=True, image_type_brats=args.image_type_brats, args=args)
    elif 'mrnet' in args.dataset_name:
        test_set = TB_Dataset_mrnet('test', use_path, False, ifbalanceloader, 'ori', False, num_classes, ifonline_aug=False, val=True, args=args)
    else:
        print('wrong dataset_name!')

    dataloaders = DataLoader(test_set, batch_size=1, num_workers=args.num_workers)

    if args.dataset_name == 'men':

        all_preds = []  # 记录每个病例的模型预测结果
        all_preds_score = []
        labels = []  # 记录每个病例的侵袭和分级标签

    elif 'brats' in args.dataset_name or 'mrnet' in args.dataset_name:
        correct = 0
        TP = 0
        FP = 0
        FN = 0
        TN = 0

        y_score = []
        y_true = []

        preds = []  # 记录每个病例的模型预测结果

        labels = []  # 记录每个病例的分类标签

    with torch.no_grad():
        for x, y, patient in dataloaders:

            x = x.type(torch.FloatTensor)
            x = x.cuda()  # if voting,[1,aug_num,2,128,128,24]; else,[2,128,128,24]
            y = y.cuda()
            labels.append(y.cpu().numpy()[0])

            if args.method_name == 'nestedFormer':
                x = x.permute([0,1,4,2,3])

            if args.method_name == 'CFDL':
                out = model(x, run_type='test')
                out_pred = out['out_prediction']
            else:
                print('wrong method_name!')
                break

            if args.dataset_name == 'men':
                out_pred = F.softmax(out_pred, dim=1)
                pred = out_pred.argmax(dim=1, keepdim=False)
                all_preds.append(pred.cpu().numpy()[0])
                all_preds_score.append(out_pred.cpu().numpy()[0])
            elif 'brats' in args.dataset_name or 'mrnet' in args.dataset_name:
                out_pred = F.softmax(out_pred, dim=1)
                pred = out_pred.argmax(dim=1, keepdim=True)
                y_score.append(out_pred[0,1].cpu().numpy())
                preds.append(pred.cpu().numpy()[0])
                y_true.append(y.cpu().numpy())

                correct += pred.eq(y.view_as(pred)).sum().item()
                if pred.view_as(y) == y == 1: TP += 1
                if (pred.view_as(y) == 1) and (y == 0): FP += 1
                if (pred.view_as(y) == 0) and (y == 1): FN += 1
                if pred.view_as(y) == y == 0: TN += 1

    if args.dataset_name == 'men':
        right_0 = 0
        right_1 = 0
        right_2 = 0
        preds = np.array(all_preds)

        for i in range(preds.shape[0]):
            if preds[i] == labels[i] and preds[i] == 0:
                right_0 += 1
            elif preds[i] == labels[i] and preds[i] == 1:
                right_1 += 1
            elif preds[i] == labels[i] and preds[i] == 2:
                right_2 += 1
        num_0 = len([i for i, x in enumerate(labels) if x == 0])
        num_1 = len([i for i, x in enumerate(labels) if x == 1])
        num_2 = len([i for i, x in enumerate(labels) if x == 2])

        all_preds_score_np = np.array(all_preds_score)
        if not np.isfinite(all_preds_score_np).all():
            print(all_preds_score_np)

        acc_g1 = Decimal(right_0 / num_0).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc_g2inv = Decimal(right_1 / num_1).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc_g2noninv = Decimal(right_2 / num_2).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc = Decimal(accuracy_score(labels,all_preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        weighted_f1 = Decimal(f1_score(labels,all_preds, average='weighted')).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        macro_f1 = Decimal(f1_score(labels,all_preds, average='macro')).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        auc = Decimal(roc_auc_score(one_hot_forauc(labels, num_classes), all_preds_score)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)

        mean_metrics = (acc+acc_g1+acc_g2inv+acc_g2noninv+weighted_f1+macro_f1+auc)/6
        out_show = '(acc:{}, acc_g1:{}, acc_g2inv:{}, acc_g2noninv:{}, weighted_f1:{}, macro_f1:{}, auc:{})'.format(str(acc),str(acc_g1),str(acc_g2inv),str(acc_g2noninv),str(weighted_f1),str(macro_f1),str(auc))
        return {'mean_metrics': mean_metrics,
                'out_show': out_show,
                'acc_g1':acc_g1,
                'acc_g2inv':acc_g2inv,
                'acc_g2noninv':acc_g2noninv,
                'acc':acc,
                'weighted_f1':weighted_f1,
                'macro_f1':macro_f1,
                'auc':auc}

    elif 'brats' in args.dataset_name or 'mrnet' in args.dataset_name:
        fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=None, sample_weight=None, drop_intermediate=True)

        sen = Decimal(TP / (TP + FN)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        spe = Decimal(TN / (TN + FP)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        acc = Decimal((TP + TN) / (TP + FP + FN + TN)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        g_mean = Decimal(((TP / (TP + FN)) * (TN / (TN + FP))) ** 0.5).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        balance_acc = Decimal(balanced_accuracy_score(labels, preds)).quantize(Decimal("0.0000"),                                                                         rounding=ROUND_HALF_UP)
        auprc = Decimal(average_precision_score(labels, preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        auc = Decimal(AUC(fpr, tpr)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)
        mcc = Decimal(matthews_corrcoef(labels, preds)).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP)

        mean_metrics = (sen + spe + acc + auprc + auc + balance_acc) / 6
        out_show = '(sen:{}, spe:{}, acc:{}, g_mean:{}, balance_acc:{}, auprc:{}, auc:{})'.format(str(sen),str(spe),str(acc),str(g_mean),str(balance_acc),str(auprc),str(auc))
        return {'mean_metrics': mean_metrics,
                'out_show': out_show,
                'sen': sen,
                'spe': spe,
                'acc': acc,
                'g_mean': g_mean,
                'balance_acc': balance_acc,
                'auprc': auprc,
                'auc': auc}


def train_model_CFDL(model, optimizer, train_sampler, dataload, scheduler, scheduler_warmup, num_epochs, loss_type):
    step_ls = []

    start_time = datetime.datetime.now()
    if args.multi_gpus:
        if local_rank == 0:
            print('epoch start time: ', start_time)
    else:
        print('epoch start time: ', start_time)

    if loss_type == 'ce':
        cls_loss_info = 'cls_loss_type:{}'.format(loss_type)
    elif loss_type == 'focal':
        cls_loss_info = 'cls_loss_type:{}-(alpha:{},gamma:{})'.format(loss_type, str(args.focal_alpha), str(args.focal_gamma))
    loss_info = cls_loss_info + ', sim_loss_type:{}, diff_loss_type:{}, w_main_cls:{}, w_sim:{}, w_diff:{}'.format(
                args.sim_loss_type, args.diff_loss_type, str(args.w_main_cls), str(args.w_sim), str(args.w_diff))


    if args.fusion_type == 'concat':
        fusion_info = 'fusion_type: {}'.format(args.fusion_type)
    elif args.fusion_type == 'moe':
        if args.moe_fusion_type == 'concat':
            fusion_info = 'fusion_type: {}, moe_fusion_type:{}, gating_type:{}, gating_concat_mlp_layer:{}, ifmoe_aux:{}, w_moe_aux:{}'.format(args.fusion_type, args.moe_fusion_type, args.gating_type, args.gating_concat_mlp_layer, args.ifmoe_aux, args.w_moe_aux)
        else:
            fusion_info = 'fusion_type: {}, moe_fusion_type:{}, gating_type:{}, ifmoe_aux:{}, w_moe_aux:{}'.format(args.fusion_type, args.moe_fusion_type, args.gating_type, args.ifmoe_aux, args.w_moe_aux)

    train_info = 'dataset:{}, method_name:{}, ifwarmup:{}, seed:{}, multi_gpus:{}, bach_size:{}, ifbalanceloader:{}, if_offline_data_aug:{}, ifonline_aug:{}, cls_drop:{}, start_lr:{}, weight_decay:{}, latent_feature_dim:{}, loss_info:{}, fusion_info:{}, ifdis_sup:{}, if_allpair:{}'.format(
                 args.dataset_name, args.method_name, str(args.ifwarmup), str(args.seed), str(args.multi_gpus),
                 str(batch_size), str(args.ifbalanceloader), str(args.if_offline_data_aug), str(1 - args.if_offline_data_aug), str(args.cls_drop),
                 str(train_lr), str(optimizer.state_dict()['param_groups'][0]['weight_decay']), str(args.feature_dim),
                 loss_info, fusion_info, str(args.ifdis_sup), str(args.if_allpair))

    if args.multi_gpus:
        if local_rank == 0:
            train_info = 'multi-gpu  ,' + train_info
    else:

        train_info = 'single-gpu  ,' + train_info

    step_ls.append(train_info)
    step_ls.append(use_path)
    step_ls.append(start_time)

    if args.multi_gpus:
        if local_rank == 0:
            weight_path = os.path.join('../weights', args.dataset_name,
                                       args.method_name, str(str(start_time).split(' ')[0]) + '-' +
                                       str((str(start_time).split(' ')[1]).split('.')[0]).split(':')[0] + '_' +
                                       str((str(start_time).split(' ')[1]).split('.')[0]).split(':')[1], str(fold))
            if not os.path.exists(weight_path):
                os.makedirs(weight_path)
    else:
        time_path = '{}-{}_{}-{}'.format(str(str(start_time).split(' ')[0]), str((str(start_time).split(' ')[1]).split(':')[0]), str((str(start_time).split(' ')[1]).split(':')[1]), str((str(start_time).split(' ')[1]).split(':')[2]))
        if 'mrnet' in args.dataset_name:
            weight_path = os.path.join('../weights', args.dataset_name, args.method_name, time_path)
        else:
            weight_path = os.path.join('../weights', args.dataset_name, args.method_name, time_path, str(fold))

        if not os.path.exists(weight_path):
            os.makedirs(weight_path)

    best_val_result = 0
    mean_metrics = 0
    for epoch in range(num_epochs):
        model.train() # 每个epoch都会进行model.eval(), 因此需要每个epoch加上model.train()

        if args.pretrained:
            show_param_groups = 1
        else:
            show_param_groups = 0

        if args.multi_gpus:
            train_sampler.set_epoch(epoch) # shuffle
            if local_rank == 0:
                print('\nEpoch {}/{}, lr:{}, time:{}\n{}'.format(str(epoch+1), str(num_epochs), str(optimizer.state_dict()['param_groups'][0]['lr']), str(datetime.datetime.now()), '-' * 60))
        else:
            print('\nEpoch {}/{}, lr:{}, time:{}\n{}'.format(str(epoch+1), str(num_epochs), str(optimizer.state_dict()['param_groups'][0]['lr']), str(datetime.datetime.now()), '-' * 60))

        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0

        for inputs, labels, patient in dataload:

            step += 1

            inputs = inputs.type(torch.FloatTensor)
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward
            # out = model(inputs, label=labels, global_step=epoch+1)
            out = model(inputs, labels, epoch+1)

            loss = out['loss']
            out_pred = out['out_prediction']

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
            # if not args.uncertainty:
            out_pred = F.softmax(out_pred, dim=1)
            pred = out_pred.argmax(dim=1, keepdim=True)
            correct = pred.eq(labels.view_as(pred)).sum().item()

            if args.dataset_name == 'men':
                if step % display_num == 0:
                    right_0 = 0
                    right_1 = 0
                    right_2 = 0
                    for i in range(pred.shape[0]):
                        if pred[i] == labels[i] and pred[i] == 0:
                            right_0 += 1
                        elif pred[i] == labels[i] and pred[i] == 1:
                            right_1 += 1
                        elif pred[i] == labels[i] and pred[i] == 2:
                            right_2 += 1
                    num_0 = len([i for i, x in enumerate(labels) if x == 0])
                    num_1 = len([i for i, x in enumerate(labels) if x == 1])
                    num_2 = len([i for i, x in enumerate(labels) if x == 2])

                    # if dif_weight_type == 'tmc':
                    #     display_data = ' || loss:{} ( main_cls_loss:{}, tmc_loss:{}, l1_norm_loss:{}, con_loss:{} ) ||  Acc: {}/{} ({:.0f}%)'.format(str(loss.item()), str(out['main_cls_loss'].item()), str(out['tmc_loss'].item()), str(out['l1_norm_loss'].item()), str(out['con_loss'].item()), correct, batch_size, 100. * correct / batch_size) +' [ 0: ' + str(right_0) + '/' + str(num_0) + ', 1: ' + str(right_1) + '/' + str(num_1) +', 2: ' + str(right_2) + '/' + str(num_2) + ']'
                    # else:
                    #     display_data = ' || loss:{} ||  Acc: {}/{} ({:.0f}%)'.format(str(loss.item()), correct, batch_size, 100. * correct / batch_size) +' [ 0: ' + str(right_0) + '/' + str(num_0) + ', 1: ' + str(right_1) + '/' + str(num_1) +', 2: ' + str(right_2) + '/' + str(num_2) + ']'
                    if args.multi_gpus:
                        if local_rank == 0:
                            if args.method_name == 'misa':
                                display_data = ' || loss:{} (main_cls_loss:{}, cmd_loss:{}, recon_loss:{}, diff_loss:{}, whiten_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()), str(out['cmd_loss'].item()),
                                    str(out['recon_loss'].item()), str(out['diff_loss'].item()), str(out['whiten_loss'].item()), correct, batch_size,
                                    100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1),
                                    str(right_2), str(num_2))
                            else:
                                display_data = ' || loss:{} (main_cls_loss:{}, aux_share_cls_loss:{}, con_loss:{}, recon_loss:{}, diff_loss:{}, whiten_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(str(loss.item()), str(out['main_cls_loss'].item()), str(out['aux_share_cls_loss'].item()),str(out['con_loss'].item()), str(out['recon_loss'].item()), str(out['diff_loss'].item()), str(out['whiten_loss'].item()), correct, batch_size,
                                    100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1), str(right_2), str(num_2))

                            print(' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

                            step_ls.append('epoch:' + str(epoch+1) + ', lr: ' + str(optimizer.state_dict()['param_groups'][show_param_groups]['lr']) + str(step) + '/' + str(
                                (dt_size - 1) // dataload.batch_size + 1) + display_data)
                    else:
                        if args.method_name == 'misa':
                            display_data = ' || loss:{} (main_cls_loss:{}, cmd_loss:{}, recon_loss:{}, diff_loss:{}, whiten_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                                str(loss.item()), str(out['main_cls_loss'].item()), str(out['cmd_loss'].item()),
                                str(out['recon_loss'].item()), str(out['diff_loss'].item()),
                                str(out['whiten_loss'].item()), correct, batch_size,
                                100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1),
                                str(right_2), str(num_2))
                        elif args.method_name == 'CCML':
                            display_data = ' || loss:{} || Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                                str(loss.item()),  correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1), str(right_2), str(num_2))
                        elif args.method_name == 'GLoMo':
                            display_data = ' || loss:{} (main_cls_loss:{}, moe_loss:{}) || Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                                str(loss.item()), str(out['main_cls_loss'].item()), str(out['moe_loss'].item()), correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1), str(right_2), str(num_2))
                        else:
                            display_data = ' || loss:{} (main_cls_loss:{}, share_sim_loss:{}, sup_sim_loss:{}, diff_loss:{}, moe_aux_loss:{}, cc_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}, 2:{}/{}]'.format(
                                str(loss.item()), str(out['main_cls_loss'].item()),
                                str(out['share_sim_loss'].item()), str(out['sup_sim_loss'].item()),
                                str(out['diff_loss'].item()), str(out['moe_aux_loss'].item()),
                                str(out['cc_loss'].item()), correct, batch_size, 100. * correct / batch_size,
                                str(right_0), str(num_0), str(right_1), str(num_1), str(right_2), str(num_2))

                        print(' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

                        step_ls.append('epoch:' + str(epoch + 1) + ', lr: ' + str(
                            optimizer.state_dict()['param_groups'][show_param_groups]['lr']) + str(step) + '/' + str(
                            (dt_size - 1) // dataload.batch_size + 1) + display_data)

            elif 'brats' in args.dataset_name or 'mrnet' in args.dataset_name:

                if step % display_num == 0:

                    new_label = labels.view_as(pred)
                    right_0 = 0
                    right_1 = 0
                    for i in range(pred.shape[0]):
                        if pred[i, 0] == new_label[i, 0] and pred[i, 0] == 1:
                            right_1 += 1
                        elif pred[i, 0] == new_label[i, 0] and pred[i, 0] == 0:
                            right_0 += 1
                    num_0 = len([i for i, x in enumerate(labels) if x == 0])
                    num_1 = len([i for i, x in enumerate(labels) if x == 1])
                    if args.multi_gpus:
                        if local_rank == 0:
                            if 'brats' in args.dataset_name:
                                display_data = ' || loss:{} ( main_cls_loss:{}, aux_share_cls_loss:{}, con_loss:{}, recon_loss:{}, diff_loss:{}, whiten_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()), str(out['aux_share_cls_loss'].item()), str(out['con_loss'].item()), str(out['recon_loss'].item()), str(out['diff_loss'].item()), str(out['whiten_loss'].item()), correct, batch_size,
                                    100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))
                            elif 'mrnet' in args.dataset_name:
                                display_data = ' || loss:{} ( main_cls_loss:{}, con_loss:{}, recon_loss:{}, diff_loss:{}, , whiten_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()), str(out['con_loss'].item()),str(out['recon_loss'].item()), str(out['diff_loss'].item()), str(out['whiten_loss'].item()), correct,
                                    batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))

                            print(' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

                            step_ls.append('epoch:' + str(epoch+1) + ', lr: ' + str(optimizer.state_dict()['param_groups'][show_param_groups]['lr']) + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)
                    else:
                        if 'brats' in args.dataset_name:
                            if args.method_name == 'CCML':
                                display_data = ' || loss:{} ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))
                            elif args.method_name == 'GLoMo':
                                display_data = ' || loss:{} (main_cls_loss:{}, moe_loss:{}) || Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()), str(out['moe_loss'].item()), correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))
                            else:
                                display_data = ' || loss:{} ( main_cls_loss:{}, share_sim_loss:{}, sup_sim_loss:{}, diff_loss:{}, cc_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()),
                                    str(out['share_sim_loss'].item()), str(out['sup_sim_loss'].item()),
                                    str(out['diff_loss'].item()), str(out['cc_loss'].item()),
                                    correct, batch_size, 100. * correct / batch_size,
                                    str(right_0), str(num_0), str(right_1), str(num_1))
                        elif 'mrnet' in args.dataset_name:
                            if args.method_name == 'CCML':
                                display_data = ' || loss:{} ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))
                            elif args.method_name == 'GLoMo':
                                display_data = ' || loss:{} (main_cls_loss:{}, moe_loss:{}) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()), str(out['moe_loss'].item()), correct, batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1), str(num_1))
                            else:
                                display_data = ' || loss:{} ( main_cls_loss:{}, share_sim_loss:{}, sup_sim_loss:{}, diff_loss:{}, cc_loss:{} ) ||  Acc: {}/{} ({:.0f}%) [0:{}/{}, 1:{}/{}]'.format(
                                    str(loss.item()), str(out['main_cls_loss'].item()),
                                    str(out['share_sim_loss'].item()), str(out['sup_sim_loss'].item()),
                                    str(out['diff_loss'].item()), str(out['cc_loss'].item()), correct,
                                    batch_size, 100. * correct / batch_size, str(right_0), str(num_0), str(right_1),
                                    str(num_1))

                        print(' ' * 5 + str(step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

                        step_ls.append('epoch:' + str(epoch + 1) + ', lr: ' + str(
                            optimizer.state_dict()['param_groups'][show_param_groups]['lr']) + str(
                            step) + '/' + str((dt_size - 1) // dataload.batch_size + 1) + display_data)

        if args.multi_gpus:
            if local_rank == 0:
                print(' ' * 5 + 'epoch ' + str(epoch + 1), ': loss is ' + str(epoch_loss / step)+',  ',datetime.datetime.now())
        else:
            print(' ' * 5 + 'epoch ' + str(epoch + 1), ': loss is ' + str(epoch_loss / step)+',  ',datetime.datetime.now())

        val_result = val(model)
        mean = val_result['mean_metrics']
        out_show = val_result['out_show']

        if args.multi_gpus:
            if local_rank == 0:
                print(' ' * 5 + '----> ' + out_show)
                step_ls.append(' ' * 5 + '----> ' + out_show)
                if mean > mean_metrics:
                    best_val_result = val_result
                    # print(best_val_result)
                    mean_metrics = mean
                    if (epoch + 1) > args.save_epoch_start or args.dataset_name == 'brats' or 'mrnet' in args.dataset_name:
                        if 'mrnet' in args.dataset_name:
                            torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(
                                num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                            torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(
                                num_epochs) + '_' + str(batch_size) + '.pth'))
                        else:
                            torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(
                                num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                            torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(
                                num_epochs) + '_' + str(batch_size) + '.pth'))
                        print(' ' * 5 + '---->  save weight of epoch ' + str(epoch + 1) + '!')
                        step_ls.append(' ' * 5 + '---->  save weight of epoch ' + str(epoch + 1) + '!')

                if (epoch + 1) % 10 == 0 and (epoch + 1) > 70:
                    torch.save(model.state_dict(), os.path.join(weight_path, fold + '_train_weight_' + str(
                        num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
        else:
            print(' ' * 5 + '----> '+out_show)
            step_ls.append(' ' * 5 + '----> '+out_show)
            if mean > mean_metrics:
                best_val_result = val_result
                mean_metrics = mean
                if (epoch + 1) > args.save_epoch_start or args.dataset_name == 'brats' or 'mrnet' in args.dataset_name:
                    if 'mrnet' in args.dataset_name:
                        torch.save(model.state_dict(), os.path.join(weight_path,'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                        torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(weight_path,fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                        torch.save(model.state_dict(), os.path.join(weight_path,fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '.pth'))
                    print(' ' * 5 + '---->  save weight of epoch ' + str(epoch + 1) + '!')
                    step_ls.append(' ' * 5 + '---->  save weight of epoch ' + str(epoch + 1) + '!')
            if train_epochs == 100:
                if (epoch + 1) % 10 == 0 and (epoch + 1) > 70:
                    if args.dataset_name == 'BRCA' or 'mrnet' in args.dataset_name:
                        torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(weight_path,fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
            elif train_epochs == 50:
                if (epoch + 1) % 10 == 0 and (epoch + 1) > 30:
                    if args.dataset_name == 'BRCA' or 'mrnet' in args.dataset_name:
                        torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(weight_path,fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
            elif train_epochs == 30:
                if (epoch + 1) % 10 == 0 and (epoch + 1) > 10:
                    if args.dataset_name == 'BRCA' or 'mrnet' in args.dataset_name:
                        torch.save(model.state_dict(), os.path.join(weight_path, 'train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))
                    else:
                        torch.save(model.state_dict(), os.path.join(weight_path,fold + '_train_weight_' + str(num_epochs) + '_' + str(batch_size) + '_epoch' + str(epoch + 1) + '.pth'))

        if args.ifwarmup:
            scheduler_warmup.step()
        else:
            scheduler.step()
    if args.multi_gpus:
        if local_rank == 0:
            result_ls = []
            result_ls.append(str(str(start_time).split(' ')[0]) + '-' + str((str(start_time).split(' ')[1]).split('.')[0]).split(':')[0] + '_' +str((str(start_time).split(' ')[1]).split('.')[0]).split(':')[1])
            if args.dataset_name == 'men':
                result_ls.append(best_val_result['acc'])
                result_ls.append(best_val_result['acc_g1'])
                result_ls.append(best_val_result['acc_g2inv'])
                result_ls.append(best_val_result['acc_g2noninv'])
                result_ls.append(best_val_result['weighted_f1'])
                result_ls.append(best_val_result['macro_f1'])
                result_ls.append(best_val_result['auc'])

                result_ls = np.reshape(result_ls, (1, len(result_ls)))
                if not os.path.exists(test_result_path):
                    head_ls = ['weight_date', 'acc', 'acc_g1', 'acc_g2inv', 'acc_g2noninv', 'weighted_f1', 'macro_f1', 'auc']
                    head_ls = np.reshape(head_ls, (1, len(head_ls)))
                    result_out = np.concatenate((head_ls, result_ls), axis=0)
                else:
                    result_in = pd.read_csv(test_result_path, header=None)
                    result_out = np.concatenate((result_in, result_ls), axis=0)
                result_out_pd = pd.DataFrame(result_out)
                result_out_pd.to_csv(test_result_path, header=False, index=False)
                print("Finish saving result_men.csv!")
            elif 'brats' in args.dataset_name or 'mrnet' in args.dataset_name:
                # print(best_val_result)
                result_ls.append(best_val_result['sen'])
                result_ls.append(best_val_result['spe'])
                result_ls.append(best_val_result['acc'])
                result_ls.append(best_val_result['g_mean'])
                result_ls.append(best_val_result['balance_acc'])
                # result_ls.append(Decimal(mcc).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
                result_ls.append(best_val_result['auprc'])
                result_ls.append(best_val_result['auc'])

                result_ls = np.reshape(result_ls, (1, len(result_ls)))
                if not os.path.exists(test_result_path):
                    head_ls = ['weight_date', 'Sen', 'Spe', 'Acc', 'G_mean', 'Ba_Acc', 'auprc', 'Auc']
                    head_ls = np.reshape(head_ls, (1, len(head_ls)))
                    result_out = np.concatenate((head_ls, result_ls), axis=0)
                else:
                    result_in = pd.read_csv(test_result_path, header=None)
                    result_out = np.concatenate((result_in, result_ls), axis=0)
                result_out_pd = pd.DataFrame(result_out)
                result_out_pd.to_csv(test_result_path, header=False, index=False)
                if 'brats' in args.dataset_name:
                    print("Finish saving result_brats.csv!")
                elif 'mrnet' in args.dataset_name:
                    print("Finish saving result_mrnet.csv!")

            end_time = datetime.datetime.now()
            step_ls.append('best_val_result: '+best_val_result['out_show'])
            step_ls.append(end_time)
            step_ls_pd = pd.DataFrame(step_ls)

            if args.dataset_name == 'BRCA' or 'mrnet' in args.dataset_name:
                step_ls_pd.to_csv(os.path.join(weight_path, 'train_step_' + str(num_epochs) + '.csv'),index=False, header=False)
            else:
                step_ls_pd.to_csv(os.path.join(weight_path, fold + '_train_step_' + str(num_epochs) + '_' + str(batch_size) + '.csv'), index=False, header=False)
    else:
        result_ls = []
        result_ls.append(time_path)
        if args.dataset_name == 'men':
            result_ls.append(best_val_result['acc'])
            result_ls.append(best_val_result['acc_g1'])
            result_ls.append(best_val_result['acc_g2inv'])
            result_ls.append(best_val_result['acc_g2noninv'])
            result_ls.append(best_val_result['weighted_f1'])
            result_ls.append(best_val_result['macro_f1'])
            result_ls.append(best_val_result['auc'])

            result_ls = np.reshape(result_ls, (1, len(result_ls)))
            if not os.path.exists(test_result_path):
                head_ls = ['weight_date', 'acc', 'acc_g1', 'acc_g2inv', 'acc_g2noninv', 'weighted_f1', 'macro_f1',
                           'auc']
                head_ls = np.reshape(head_ls, (1, len(head_ls)))
                result_out = np.concatenate((head_ls, result_ls), axis=0)
            else:
                result_in = pd.read_csv(test_result_path, header=None)
                result_out = np.concatenate((result_in, result_ls), axis=0)
            result_out_pd = pd.DataFrame(result_out)
            result_out_pd.to_csv(test_result_path, header=False, index=False)
            print("Finish saving result_men.csv!")
        elif 'brats' in args.dataset_name or 'mrnet' in args.dataset_name:
            print(best_val_result)
            result_ls.append(best_val_result['sen'])
            result_ls.append(best_val_result['spe'])
            result_ls.append(best_val_result['acc'])
            result_ls.append(best_val_result['g_mean'])
            result_ls.append(best_val_result['balance_acc'])
            # result_ls.append(Decimal(mcc).quantize(Decimal("0.0000"), rounding=ROUND_HALF_UP))
            result_ls.append(best_val_result['auprc'])
            result_ls.append(best_val_result['auc'])

            result_ls = np.reshape(result_ls, (1, len(result_ls)))
            if not os.path.exists(test_result_path):
                head_ls = ['weight_date', 'Sen', 'Spe', 'Acc', 'G_mean', 'Ba_Acc', 'auprc', 'Auc']
                head_ls = np.reshape(head_ls, (1, len(head_ls)))
                result_out = np.concatenate((head_ls, result_ls), axis=0)
            else:
                result_in = pd.read_csv(test_result_path, header=None)
                result_out = np.concatenate((result_in, result_ls), axis=0)
            result_out_pd = pd.DataFrame(result_out)
            result_out_pd.to_csv(test_result_path, header=False, index=False)
            if 'brats' in args.dataset_name:
                print("Finish saving result_brats.csv!")
            elif 'mrnet' in args.dataset_name:
                print("Finish saving result_mrnet.csv!")

        end_time = datetime.datetime.now()
        step_ls.append('best_val_result: ' + best_val_result['out_show'])
        step_ls.append(end_time)
        step_ls_pd = pd.DataFrame(step_ls)

        if args.dataset_name == 'BRCA' or 'mrnet' in args.dataset_name:
            step_ls_pd.to_csv(os.path.join(weight_path, 'train_step_' + str(num_epochs) + '.csv'), index=False, header=False)
        else:
            step_ls_pd.to_csv(os.path.join(weight_path, fold + '_train_step_' + str(num_epochs) + '_' + str(batch_size) + '.csv'), index=False, header=False)


def load_weight(weight_path):
    state_dict_small = torch.load(weight_path)
    for tmp in list(state_dict_small.keys()):
        if 'linear' in tmp:
            del state_dict_small[tmp]
    return state_dict_small


def model_select(dim_list=None):
    if args.loss_type == 'ce':
        criterion = torch.nn.CrossEntropyLoss(reduction='mean')
    else:
        print('wrong loss_type!')

    if args.method_name == 'CFDL':
        model = CFDL(args=args, criterion=criterion, num_classes=num_classes, dim_list=dim_list)
    else:
        print('model_select -- wrong method_name!')

    if args.multi_gpus:
        if local_rank == 0:
            print('-'*30+'  '+args.method_name+'  '+'-'*30)
    else:
        print('-'*30+'  '+args.method_name+'  '+'-'*30)

    return model


def train(batch_size, lr, saved_weight_path):

    dim_list = None

    model = model_select(dim_list=dim_list)

    if args.finetuning:
        print('Loading weight' + '.' * 20)
        model.load_state_dict(torch.load(saved_weight_path))
    if ifdecoupletrain:
        print('Loading weight' + '.' * 20)
        model.load_state_dict(load_weight(saved_weight_path), strict=False)
        lr = lr * 0.001

        for name, value in model.named_parameters():
            if 'linear' in name:
                print(name, 'requires_grad->True')
                value.requires_grad = True
            else:
                value.requires_grad = False

    model_parameters = filter(lambda p: p.requires_grad, model.parameters())

    if args.multi_gpus:
        if local_rank == 0:
            print('Num of parameters: ', sum([np.prod(p.size()) for p in model_parameters]))

        model.to(device)
        num_gpus = torch.cuda.device_count()
        if num_gpus > 1:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)
    else:
        # print('Num of parameters: ', sum([np.prod(p.size()) for p in model_parameters]))
        print("Number of parameter: %.2fM" % (sum([np.prod(p.size()) for p in model_parameters]) / 1e6))
        model = model.cuda()

    model.train()

    if args.pretrained:
        base_parameters = []
        for pname, p in model.named_parameters():
            if 'backbone_encoder' in pname:
                base_parameters.append(p)
        print('Num of base parameters: ', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, base_parameters)]))
        base_parameters_id = list(map(id, base_parameters))
        new_parameters = list(filter(lambda p: id(p) not in base_parameters_id, model.parameters()))
        print('Num of dis parameters: ', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, new_parameters)]))
        model_params = [{"params": base_parameters, "lr": lr * args.ft_rate},
                        {"params": new_parameters, "lr": lr}]

        optimizer = optim.Adam(model_params, lr=lr, weight_decay=args.weight_decay_value)
    elif args.finetuning:
        base_parameters = []
        for pname, p in model.named_parameters():
            if 'sequential' in pname:
                base_parameters.append(p)
        print('Num of base parameters: ', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, base_parameters)]))
        base_parameters_id = list(map(id, base_parameters))
        new_parameters = list(filter(lambda p: id(p) not in base_parameters_id, model.parameters()))
        print('Num of dis parameters: ', sum([np.prod(p.size()) for p in filter(lambda p: p.requires_grad, new_parameters)]))
        model_params = [{"params": base_parameters, "lr": lr * args.ft_rate},
                        {"params": new_parameters, "lr": lr}]

        optimizer = optim.Adam(model_params, lr=lr, weight_decay=args.weight_decay_value)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=args.weight_decay_value)

    if args.dataset_name == 'men' or args.dataset_name == 'brats' or 'mrnet' in args.dataset_name:
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=args.lr_gamma) # gamma=0.8

    if args.ifwarmup:
        scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=5, after_scheduler=scheduler)
    else:
        scheduler_warmup=False

    if args.multi_gpus:
        if args.dataset_name == 'men':
            train_set = TB_Dataset_men('train', use_path, args.if_offline_data_aug, ifbalanceloader, False, image_type, num_classes, 1-args.if_offline_data_aug,args=args)
        elif 'brats' in args.dataset_name:
            train_set = TB_Dataset_brats('train', use_path, args.if_offline_data_aug, ifbalanceloader, False, False, num_classes, 1-args.if_offline_data_aug, image_type_brats=args.image_type_brats,args=args)
        elif 'mrnet' in args.dataset_name:
            train_set = TB_Dataset_mrnet('train', use_path, args.if_offline_data_aug, ifbalanceloader, False, False, num_classes, 1-args.if_offline_data_aug,args=args)
        train_sampler = DistributedSampler(train_set)

        dataloaders = DataLoader(train_set, sampler=train_sampler, batch_size=batch_size, num_workers=args.num_workers, drop_last=True, pin_memory=True)
    else:
        if args.dataset_name == 'men':
            train_set = TB_Dataset_men('train', use_path, args.if_offline_data_aug, ifbalanceloader, False, image_type, num_classes, 1-args.if_offline_data_aug, False, args)
        elif 'brats' in args.dataset_name:
            train_set = TB_Dataset_brats('train', use_path, args.if_offline_data_aug, ifbalanceloader, False, image_type, num_classes, 1-args.if_offline_data_aug, val=False, args=args)
        elif 'mrnet' in args.dataset_name:
            train_set = TB_Dataset_mrnet('train', use_path, args.if_offline_data_aug, ifbalanceloader, False, False, num_classes, 1-args.if_offline_data_aug, val=False, args=args)

        dataloaders = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=args.num_workers, drop_last=True)
        train_sampler = False


    if args.method_name == 'CFDL':
        train_model_CFDL(model, optimizer, train_sampler, dataloaders, scheduler, scheduler_warmup, train_epochs, args.loss_type)
    else:
        print('train -- wrong method_name!')


if __name__ == '__main__':
    decouple_train_lr = 0.001
    decouple_train_epochs = 50

    global modaility_num, ifdecoupletrain, is_training
    global image_type, use_path, test_result_path

    image_type = 'bbox'
    ifdecoupletrain = False

    if args.dataset_name == 'men':
        test_result_path = '../weights/result_men.csv'
    elif 'brats' in args.dataset_name:
        test_result_path = '../weights/result_brats.csv'
    elif args.dataset_name == 'mrnet-meniscus':
        test_result_path = '../weights/result_mrnet-meniscus.csv'


    if args.dataset_name == 'men':
        data_path = 'CFDL_data/men/cv3folders_multicls'
    elif args.dataset_name == 'brats':
        data_path = 'CFDL_data/brats/cv3folder'
    elif args.dataset_name == 'mrnet-meniscus':
        data_path = 'CFDL_data/mrnet/MRNet-meniscus'

    if args.multi_gpus:
        if local_rank == 0:
            print('start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))
    else:
        print('start time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

    if args.run_type == 'train':
        weight_path = False

        if args.multi_gpus:
            if local_rank == 0:
                if args.dataset_name == 'men':
                    print(args.run_type + ' <<<   epoch: ' + str(train_epochs) + '   lr: ' + str(train_lr) + '   batch_size: ' + str(
                        batch_size) + '   weight_decay:' + str(args.weight_decay_value) + '   cls_drop:' + str(args.cls_drop))
                elif 'brats' in args.dataset_name:
                    print(args.run_type + ' <<<   epoch: ' + str(train_epochs) + '   lr: ' + str(train_lr) + '   batch_size: ' + str(
                        batch_size) + '   weight_decay:' + str(args.weight_decay_value) + '   cls_drop:' + str(args.cls_drop))
        else:
            if args.dataset_name == 'men' or 'mrnet' in args.dataset_name:
                print(args.run_type + ' <<<   epoch: ' + str(train_epochs) + '   lr: ' + str(train_lr) + '   batch_size: ' + str(
                    batch_size) + '   weight_decay:' + str(args.weight_decay_value) + '   cls_drop:' + str(args.cls_drop))
            elif 'brats' in args.dataset_name:
                print(args.run_type + ' <<<   epoch: ' + str(train_epochs) + '   lr: ' + str(train_lr) + '   batch_size: ' + str(
                        batch_size) + '   weight_decay:' + str(args.weight_decay_value) + '   drop:' + str(args.cls_drop))

        if args.fold_train == 1:
            fold_list = ['1fold']
        elif args.fold_train == 2:
            fold_list = ['2fold']
        elif args.fold_train == 3:
            fold_list = ['3fold']
        elif args.fold_train == 12:
            fold_list = ['1fold','2fold']
        elif args.fold_train == 13:
            fold_list = ['1fold','3fold']
        elif args.fold_train == 23:
            fold_list = ['2fold','3fold']
        elif args.fold_train == 123:
            fold_list = ['1fold','2fold','3fold']
        else:
            print('wrong fold_train')

        if 'mrnet' in args.dataset_name:
            use_path = data_path
            train(batch_size, train_lr, weight_path)
        else:
            for fold in fold_list:
                if args.multi_gpus:
                    if local_rank == 0:
                        print('processing------' + fold)
                else:
                    print('processing------'+fold)
                use_path = os.path.join(data_path, fold)
                train(batch_size, train_lr, weight_path)

    else:
        print('Wrong run_type!')
    print('\nend time:', time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())))

