import torch
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data.distributed
from tqdm import tqdm
import gc
import copy
import pickle
import numpy as np
from Models import Model
from torch.utils.data import Dataset, DataLoader
from data_set import All_Client_Dataset
from functools import reduce
from utils import AverageMeter
import torch.backends.cudnn as cudnn
from itertools import chain
from tqdm import tqdm
from tqdm import trange
import shutil
import time


class Client(object):
    def __init__(self, cid, args, data, cname, pre=False):
        self.cid = cid
        self.c_name = cname
        self.args = args
        self.pos_num = args.pos_num
        self.test_feat = {}
        self.val_feat = {}
        self.train_feat = {}
        self.global_round = -1
        self.num_classes = data.train_class_sizes[self.cid]
        self.loss = 0
        self.best_acc = 0
        if self.c_name == 'FP':
            self.local_epoch = args.pr_local_epoch
        else:
            self.local_epoch = args.ve_local_epoch
        self.dataset_size = data.train_dataset_sizes[self.cid]
        self.train_loader = data.train_loaders[self.cid]
        self.val_loader = data.val_loaders[self.cid]
        # The global base ID for each client (ex. local: 0-99, global: 300-399 )
        # self.ID_base = data.train_loaders[self.cid].dataset.ID_base
        # self.target_ID = list(range(self.ID_base,self.ID_base+self.num_classes))

        if hasattr(data, 'test_loaders'):
            self.test_loader = data.test_loaders[self.cid]
        if hasattr(data, 'public_train_loader'):
            self.public_num_classes = self.args.pretrained_num

        ### Create directory
        self.client_output = os.path.join(args.save_dir, self.c_name)
        if not os.path.exists(self.client_output):
            os.mkdir(self.client_output)
        self.client_output = os.path.join(self.client_output, 'client_%d' % (self.cid))
        if not os.path.exists(self.client_output):
            os.mkdir(self.client_output)

        if pre == True:
            self.pre_loader = data.public_train_loader
            self.federated_model = Model(args.network, args.pretrained_num, args.pretrained_root)

        ### FC module, on cpu
        self.model = Model(args.network, self.num_classes, self.client_output)

    def Fed_Link(self, clients):
        self.clients = clients

    def pre_train(self, Epoch):
        cudnn.benchmark = True

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        train_loader = self.pre_loader

        pre_model = self.federated_model.model
        pre_model.train()
        pre_fc = self.federated_model.fc
        pre_fc.train()

        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.SGD(params=chain(pre_model.parameters(), pre_fc.parameters()), lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)
        #optimizer_fc = torch.optim.SGD(pre_fc.parameters(), self.args.lr,
        #momentum=self.args.momentum,
        #weight_decay=self.args.weight_decay)
        pre_model = pre_model.cuda()
        pre_fc = pre_fc.cuda()

        end = time.time()

        with trange(Epoch) as t:
            for epo in t:
                for i, (input, target) in enumerate(train_loader):
                    # measure data loading time
                    data_time.update(time.time() - end)
                    input = input.cuda()
                    target = target.cuda()
                    input_var = torch.autograd.Variable(input)
                    target_var = torch.autograd.Variable(target)

                    # compute output
                    output = pre_model(input_var)
                    output = pre_fc(output)
                    loss = criterion(output, target_var)

                    # measure accuracy and record loss
                    prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
                    losses.update(loss.item(), input.size(0))
                    top1.update(prec1.item(), input.size(0))
                    top5.update(prec5.item(), input.size(0))

                    # compute gradient and do SGD step
                    optimizer.zero_grad()
                    #optimizer_fc.zero_grad()
                    loss.backward()
                    optimizer.step()
                    #optimizer_fc.step()

                    # measure elapsed time
                    batch_time.update(time.time() - end)
                    end = time.time()

                # 设置进度条左边显示的信息
                t.set_description("Epoch %i" % epo)
                # 设置进度条右边显示的信息
                t.set_postfix(loss=losses.avg, top1=top1.avg, top5=top5.avg)

        print('Serve_{c_name}_pretrain'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            c_name=self.c_name, loss=losses, top1=top1, top5=top5))

        print('Finish ' + self.c_name + ' Pre_Train')
        if not os.path.exists(self.client_output):
            os.mkdir(self.client_output)
        torch.save({'state_dict': pre_model.state_dict()},
                   self.client_output + '/' + self.c_name + '_pre_model.pth.tar')

    def train(self):
        # put model to gpu
        cudnn.benchmark = True

        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()

        train_loader = self.train_loader

        local_model, local_fc = self.model.load_model()
        local_model.train()
        local_fc.train()

        criterion = nn.CrossEntropyLoss().cuda()

        optimizer = torch.optim.SGD(params=chain(local_model.parameters(), local_fc.parameters()), lr=self.args.lr,
                                    momentum=self.args.momentum,
                                    weight_decay=self.args.weight_decay)

        local_model = local_model.cuda()
        local_fc = local_fc.cuda()

        end = time.time()

        for epo in range(self.local_epoch):
            for i, (input, target) in enumerate(train_loader):
                # measure data loading time
                data_time.update(time.time() - end)
                input = input.cuda()
                target = target.cuda()
                input_var = torch.autograd.Variable(input)
                target_var = torch.autograd.Variable(target)

                # compute output
                output = local_model(input_var)
                output = local_fc(output)
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

        local_model.cpu()
        local_fc.cpu()

        print('[{c_name}]Client_Train[{cid}]'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            c_name=self.c_name, cid=self.cid, loss=losses, top1=top1, top5=top5))

        if not os.path.exists(self.client_output):
            os.mkdir(self.client_output)
        torch.save({'state_dict': local_model.state_dict()}, self.client_output + '/local_model.pth.tar')
        torch.save({'state_dict': local_fc.state_dict()}, self.client_output + '/local_fc.pth.tar')

    def test(self, is_fate=False, feat_data='test'):
        # put model to gpu
        batch_time = AverageMeter()
        data_time = AverageMeter()
        losses = AverageMeter()
        top1 = AverageMeter()
        top5 = AverageMeter()
        if feat_data == 'train':
            test_loader = self.train_loader
        else:
            test_loader = self.test_loader
        if is_fate:
            id_dict = {}
            for key, val in test_loader.dataset.class_to_idx.items():
                id_dict[val] = key

        local_model, local_fc = self.model.load_model()
        local_model.eval()
        local_fc.eval()

        criterion = nn.CrossEntropyLoss().cuda()

        local_model = local_model.cuda()
        local_fc = local_fc.cuda()

        end = time.time()
        for i, (input, target) in enumerate(test_loader):
            # measure data loading time
            data_time.update(time.time() - end)
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)
            target_var = torch.autograd.Variable(target)

            # compute output
            output = local_model(input_var)
            lable = target.data.cpu().numpy()
            if is_fate:
                for i, feat in enumerate(output.data):
                    k = id_dict[lable[i]]
                    if feat_data == 'test':
                        if k not in self.test_feat.keys():
                            self.test_feat[id_dict[lable[i]]] = [feat]
                        else:
                            self.test_feat[id_dict[lable[i]]].append(feat)

                    else:
                        if k not in self.test_feat.keys():
                            self.train_feat[id_dict[lable[i]]] = [feat]
                        else:
                            self.train_feat[id_dict[lable[i]]].append(feat)
                continue
            output = local_fc(output)
            loss = criterion(output, target_var)

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            top5.update(prec5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

        local_model.cpu()
        local_fc.cpu()
        self.loss = round(losses.avg, 3)

        if is_fate:
            print('Finish ' + self.c_name + str(self.cid) + feat_data + ' col_fate')
            return

        print('[{c_name}]Client_Test[{cid}]'
              'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
              'Acc@1 {top1.val:.3f} ({top1.avg:.3f})\t'
              'Acc@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
            c_name=self.c_name, cid=self.cid, loss=losses, top1=top1, top5=top5))

        if top1.avg > self.best_acc:
            self.best_acc = top1.avg

    def despatch(self, num=20, round=-1):
        self.round = round
        for client in self.clients[:num]:
            idx = ''
            if self.round == -1:
                sever_path = self.client_output + '/' + self.c_name + '_pre_model.pth.tar'
                idx = '_pre_'
            else:
                sever_path = self.client_output + '/' + self.c_name + '_global_model.pth.tar'
                idx = '_glo_'
            local_path = client.client_output + '/local_model.pth.tar'
            shutil.copyfile(sever_path, local_path)
        print('Finish ' + self.c_name + idx + str(self.round) + ' despatch')

    def get_global_model(self, t=0.5):
        self.Col_loss()
        loss_sum = self.get_clients_losssum()
        sum_dict = {}

        for client in self.clients:
            if client.cid == self.cid:
                continue
            local_dict = client.get_model().state_dict()
            weight = self.loss_list[client.cid] / loss_sum
            weight = round(weight, 3)
            if sum_dict == {}:
                for key, var in local_dict.items():
                    sum_dict[key] = weight * var.clone()
            else:
                for key in local_dict.keys():
                    sum_dict[key] = sum_dict[key] + weight * local_dict[key]

        local_dict = self.get_model().state_dict()
        for key in local_dict.keys():
            sum_dict[key] = ((len(self.clients) - 1) / len(self.clients)) * sum_dict[key] + (1 / len(self.clients)) * \
                            local_dict[key]

        torch.save({'state_dict': sum_dict}, self.client_output + '/' + self.c_name + '_global_model.pth.tar')

    def get_global_model2(self, t=0.5):
        #self.Col_loss()
        sum_dict = {}
        global_dict = {}

        for client in self.clients:
            local_dict = client.get_model().state_dict()
            if sum_dict == {}:
                for key, var in local_dict.items():
                    sum_dict[key] = var.clone()
            else:
                for key in local_dict.keys():
                    sum_dict[key] = sum_dict[key] + local_dict[key]

        for key in sum_dict.keys():
            global_dict[key] = sum_dict[key] / 40

        torch.save({'state_dict': global_dict}, self.client_output + '/' + self.c_name + '_global_model.pth.tar')

    def get_clients_losssum(self):
        return sum(self.loss_list)

    def Cllent_valfeats(self, model):
        val_loader = self.val_loader

        model.eval()
        model.cuda()

        id_dict = {}
        val_feat = {}
        for key, val in val_loader.dataset.class_to_idx.items():
            id_dict[val] = key

        for i, (input, target) in enumerate(val_loader):
            # measure data loading time
            input = input.cuda()
            target = target.cuda()
            input_var = torch.autograd.Variable(input)

            # compute output
            output = model(input_var)
            lable = target.data.cpu().numpy()

            for i, feat in enumerate(output.data):
                k = id_dict[lable[i]]
                if k not in val_feat.keys():
                    val_feat[k] = [feat]
                else:
                    val_feat[k].append(feat)

        model.cpu()
        return val_feat

    def CosLoss_loader(self):
        self.vFRR_feat_data = []
        self.vFAR_feat_data = []

        for label, feats in self.val_feats.items():
            for i, feat in enumerate(feats):
                feat = [item.cpu().detach().numpy() for item in feat]
                j = i + 1
                while j < len(feats):
                    o_feat = [item.cpu().detach().numpy() for item in feats[j]]
                    self.vFRR_feat_data.append([feat, o_feat])
                    j += 1

                for ot_label, ot_feats in self.val_feats.items():
                    if label >= ot_label:
                        continue
                    #rod = random.randint(0,len(ot_feats)-1)
                    for ot_feat in ot_feats:
                        ot_feat = [item.cpu().detach().numpy() for item in ot_feat]
                        self.vFAR_feat_data.append([feat, ot_feat])

    def Col_F(self):
        self.vFAR_cosdis = []
        self.vFRR_cosdis = []
        for far in self.vFAR_feat_data:
            cos = np.dot(far[0], far[1]) / (np.linalg.norm(far[0]) * np.linalg.norm(far[1]))
            self.vFAR_cosdis.append(cos)
        for frr in self.vFRR_feat_data:
            cos = np.dot(frr[0], frr[1]) / (np.linalg.norm(frr[0]) * np.linalg.norm(frr[1]))
            self.vFRR_cosdis.append(cos)

    def col_each_loss(self, t):
        num = 0
        for tar in self.vFAR_cosdis:
            if tar < t:
                num += 1
        FAR = num / len(self.vFAR_feat_data)

        num = 0
        for trr in self.vFRR_cosdis:
            if trr > t:
                num += 1
        FRR = num / len(self.vFRR_feat_data)

        return round((FAR + FRR - 1), 3)

    def Col_loss(self):
        max = 0
        self.val_feats = {}
        self.loss_list = []
        for client in tqdm(self.clients, desc='Col_val_loss'):
            if client.cid == self.cid:
                self.loss_list.append(0)
                continue
            val_model = client.get_model()
            self.val_feats = self.Cllent_valfeats(val_model)
            self.CosLoss_loader()
            self.Col_F()
            loss = self.col_each_loss(t=0.9)
            if loss > max:
                max = loss
                self.next_center = client.cid
            self.loss_list.append(loss)

    def get_model(self):
        model, _ = self.model.load_model()
        return model


class wick_Client(object):
    def __init__(self, cid, cname, args):
        self.cid = cid
        self.c_name = cname
        self.num_classes = 10
        self.client_output = os.path.join(args.save_dir, self.c_name)
        if not os.path.exists(self.client_output):
            os.mkdir(self.client_output)
        self.client_output = os.path.join(self.client_output, 'client_%d' % (self.cid))
        if not os.path.exists(self.client_output):
            os.mkdir(self.client_output)
        self.model = Model(args.network, self.num_classes, self.client_output)

    def get_model(self):
        return self.model.model

    def Fed_Link(self, clients):
        self.clients = clients

    def despatch(self, num=20, round=-1):
        self.round = round
        for client in self.clients:
            idx = ''
            if client.cid == self.cid:
                continue
            if self.round == -1:
                sever_path = self.client_output + '/' + self.c_name + '_pre_model.pth.tar'
                idx = '_pre_'
            else:
                sever_path = self.client_output + '/' + self.c_name + '_global_model.pth.tar'
                idx = '_glo_'
            local_path = client.client_output + '/local_model.pth.tar'
            shutil.copyfile(sever_path, local_path)
        print('Finish ' + self.c_name + idx + '_' + str(self.round) + ' despatch')


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res
