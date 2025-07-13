import os
from tqdm import tqdm
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import optim
from Models import Model, Muti_Model
from client import Client
from utils import AverageMeter
import torchvision.transforms as transforms
import time
import torch.utils.data as Data
import shutil
import torch.backends.cudnn as cudnn
from itertools import chain
from tqdm import tqdm
from tqdm import trange
import random

def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class Server(object):
    def __init__(self, FL_List, args):
        self.args = args
        self.FL_List = FL_List
        self.FAR_cosdis = []
        self.FRR_cosdis = []
        self.FAR_dis = []
        self.FRR_dis = []
        self.linear = Muti_Model()
        self.mul_feat = {}


    def Collent_feat(self,feat_mode = 'train',num = 20):
        if feat_mode == 'train':
            for FL in self.FL_List:
                if FL.name == 'FP':
                    self.linear.print_dic2 = FL.linear.print_dic2
                else:
                    self.linear.vein_dic2 = FL.linear.vein_dic2
        else:
            for FL in self.FL_List:
                if FL.name == 'FP':
                    self.linear.print_dic = FL.linear.print_dic
                else:
                    self.linear.vein_dic = FL.linear.vein_dic

        self.linear.co_label(feat_mode)

    def Feat_Data(self, feat_data):
        if feat_data == 'train':
            label_list = [(int(data[0])-1) for data in self.linear.train_data_loder]
            data_list = [data[1] for data in self.linear.train_data_loder]
        else:
            label_list = [(int(data[0]) - 1) for data in self.linear.test_data_loder]
            data_list = [data[1] for data in self.linear.test_data_loder]
        return label_list,data_list

    def mul_train(self, Epoch, is_feat = False, mode = 'train'):
        label_list, data_list = self.Feat_Data(feat_data=mode)
        label_list = torch.tensor(label_list).cpu()
        data_list = torch.tensor([item.cpu().detach().numpy() for item in data_list]).cuda()
        data_list.cpu()
        data_set = Data.TensorDataset(data_list, label_list)
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        cudnn.benchmark = True

        train_loader = torch.utils.data.DataLoader(data_set,
                batch_size=self.args.batch_size, shuffle=True, pin_memory=False)

        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.SGD(params=chain(self.linear.mul_fc.parameters(), self.linear.fc.parameters()), lr=0.001,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        self.linear.mul_fc.cuda()
        self.linear.fc.cuda()

        if is_feat:
            self.linear.mul_fc.eval()
            self.linear.fc.eval()
        end = time.time()

        for epoch in range(Epoch):
            for i, (input, target) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                input = input.cuda()
                target = target.cuda()
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                feats = self.linear.mul_fc(input_var)
                if is_feat:
                    lable = target.data.cpu().numpy()
                    feat_list = feats.data.cpu().numpy()
                    for i, feat in enumerate(feat_list):
                        if lable[i] not in self.mul_feat.keys():
                            self.mul_feat[lable[i]] = [feat]
                        else:
                            self.mul_feat[lable[i]].append(feat)
                    continue

                output = self.linear.fc(feats)
                loss = criterion(output, target_var)

                # measure accuracy and record loss
                prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                losses.update(loss.item(), input.size(0))
                top1.update(prec1.item(), input.size(0))
                top5.update(prec5.item(), input.size(0))

                # compute gradient and do SGD step
                optimizer.zero_grad()
                # optimizer_fc.zero_grad()
                loss.backward()
                optimizer.step()
                # optimizer_fc.step()

                # measure elapsed time
                batch_time.update(time.time() - end)
                end = time.time()


        self.linear.mul_fc.cpu()
        self.linear.fc.cpu()

        if is_feat:
            print('Finish ex_feat')
            return

        print('Mulinear_Train '
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            loss=losses, top1=top1, top5=top5))

    def mul_test(self, mode = True):
        if mode == True:
            self.mul_train(Epoch=1,is_feat=True, mode = 'test')
        self.test_feat_data_load()
        self.cul_cosdistence()
        # self.cul_distence()

    def test_feat_data_load(self):
        self.FAR_feat_data = []
        self.FRR_feat_data = []

        for label, feats in tqdm(self.mul_feat.items(),desc='load_data'):
            for i, feat in enumerate(feats):
                j = i + 1
                while j < len(feats):
                    self.FRR_feat_data.append([feat,feats[j]])
                    j += 1

                for ot_label, ot_feats in self.mul_feat.items():
                    if label >= ot_label:
                        continue
                    #rod = random.randint(0,len(ot_feats)-1)
                    for ot_feat in ot_feats:
                        self.FAR_feat_data.append([feat, ot_feat])

    def cul_cosdistence(self):
        for far in tqdm(self.FAR_feat_data,desc='Cul_FAR_cos'):
            cos = np.dot(far[0], far[1]) / (np.linalg.norm(far[0]) * np.linalg.norm(far[1]))
            self.FAR_cosdis.append(cos)
        np.save('./' + 'FAR_cos', self.FAR_cosdis)
        for frr in tqdm(self.FRR_feat_data,desc='Cul_FAR_cos'):
            cos = np.dot(frr[0], frr[1]) / (np.linalg.norm(frr[0]) * np.linalg.norm(frr[1]))
            self.FRR_cosdis.append(cos)
        np.save('./' + 'FRR_cos', self.FRR_cosdis)


if __name__=="__main__":
    pass

