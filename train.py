import argparse
import logging
import os
import time
import warnings

warnings.filterwarnings("ignore")
import random
import torch
import torch.utils.data.distributed
import numpy as np
from data_set import All_Client_Dataset
import shutil
from utils import AverageMeter
from client import Client, wick_Client
from server import Server
from Federated_Link import Federated_Link
from tqdm import tqdm

parser = argparse.ArgumentParser(description='PyTorch Federated Link Training')

parser.add_argument('--test', default=False, help='batch_size per device')
parser.add_argument('--re_divide', default=True)
parser.add_argument('--is_pre', default=True)
parser.add_argument('--didata_list',
                    default=[10, 10, 20, 20, 25, 30, 30, 30, 35, 35, 40, 40, 45, 50, 50, 50, 60, 60, 90, 110])

parser.add_argument('--pretrained_num', default=840)
parser.add_argument('--pos_num', default=20)  #
parser.add_argument('--wick_num', default=20)
parser.add_argument('--pre_epoch', default=10)
parser.add_argument('--data_name', type=str, default="NUPT-FPV", help="database name")

parser.add_argument('--print_data_dir',
                    default='./data_FPV/NUPT-FPV_FULL_325/Process_gray_full_840_2class_325/FP_process_gray_2class_325',
                    help='output directory')
parser.add_argument('--vein_data_dir',
                    default='./data_FPV/NUPT-FPV_FULL_325/Process_gray_full_840_2class_325/FV_process_gray_2class_325',
                    help='output directory')
parser.add_argument('--pretrained_root', default='./c_pth')
parser.add_argument('--FP_ex_pretrained_root',
                    default='./data_FPV/NUPT-FPV_FULL_325/Process_gray_full_840_1class_325/FP_process_gray_1class_325')
parser.add_argument('--FV_ex_pretrained_root',
                    default='./data_FPV/NUPT-FPV_FULL_325/Process_gray_full_840_1class_325/FV_process_gray_1class_325')
parser.add_argument('--federate_dir', default='./GJ/federated_data', help='output directory')
parser.add_argument('--save_dir', default='./c_pth', help='output directory')
parser.add_argument('--save_pair', default='./pair', help='output directory')

parser.add_argument('--network', type=str, default="resnet18", help="net name")
parser.add_argument('--batch_size', default=8, type=int, help='batch_size per device')
parser.add_argument('--pr_local_epoch', default=1, type=int)
parser.add_argument('--ve_local_epoch', default=1, type=int)
parser.add_argument('--total_round', default=10, type=int)
parser.add_argument('--num_client', default=20, type=int)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=0.0001, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')
####
args_ = parser.parse_args()


def clean_local_pth():
    if not os.path.exists(args_.save_dir + '/FP'):
        return
    path = os.listdir(args_.save_dir + '/FP')
    for p in path:
        file_path = args_.save_dir + '/FP/' + p
        shutil.rmtree(file_path)
        os.makedirs(file_path)
    path = os.listdir(args_.save_dir + '/FV')
    for p in path:
        file_path = args_.save_dir + '/FV/' + p
        shutil.rmtree(file_path)
        os.makedirs(file_path)
    if os.path.exists('./pair'):
        shutil.rmtree('./pair')
    print('Finish clean')


def main(args):
    if not (args.test or (not args.is_pre)):
        clean_local_pth()
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    if args.re_divide:
        shutil.rmtree(args.save_dir)
        os.makedirs(args.save_dir)
        print('Redivide dataset')

    # create dataset
    FP_all_data = All_Client_Dataset(root_dir=args.print_data_dir, split_dir=args.federate_dir, args=args,
                                     ex_pretrained_root=args.FP_ex_pretrained_root, c_name='FP',
                                     re_divide=args.re_divide)
    FV_all_data = All_Client_Dataset(root_dir=args.vein_data_dir, split_dir=args.federate_dir, args=args,
                                     ex_pretrained_root=args.FV_ex_pretrained_root, c_name='FV',
                                     re_divide=args.re_divide)

    clients = []
    #x = random.randint(0,args.pos_num)
    x = 1
    for i in tqdm(range(args.num_client), desc='cerate FP_clients'):
        if i == x:
            clients.append(Client(cid=i, args=args, data=FP_all_data, cname='FP', pre=True))
        else:
            clients.append(Client(cid=i, args=args, data=FP_all_data, cname='FP'))
    for i in tqdm(range(args.wick_num), desc='create wick_FP_clients'):
        clients.append(wick_Client(cid=int(i) + args.num_client, args=args, cname='FP'))
    FL_P = Federated_Link(clients, 'FP', clients[x])

    clients = []
    for i in tqdm(range(args.num_client), desc='create FV_clients'):
        if i == x:
            clients.append(Client(cid=i, args=args, data=FV_all_data, cname='FV', pre=True))
        else:
            clients.append(Client(cid=i, args=args, data=FV_all_data, cname='FV'))
    for i in tqdm(range(args.wick_num), desc='create wick_FV_clients'):
        clients.append(wick_Client(cid=int(i) + args.num_client, args=args, cname='FV'))
    FL_V = Federated_Link(clients, 'FV', clients[x])
    print('Finish create FP_clients,FV_clients')

    Fed_Links = [FL_P, FL_V]

    for FL in Fed_Links:
        if args.is_pre:
            FL.center_client.pre_train(Epoch=args.pre_epoch)
        FL.center_client.despatch(num=args.pos_num, round=FL.round)
        FL.round += 1

    if not args.test:
        best = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(args.total_round):
            print('\n------------Epoch %i------------' % i)
            for FL in Fed_Links:
                for client in FL.clients[:args.pos_num]:
                    print('\n------------Train %d------------' % client.cid)
                    client.train()
                    print('\n------------Test %d------------' % client.cid)
                    client.test()
                    best[client.cid] = client.best_acc

                print(best)

                FL.center_client.get_global_model()
                # FL.center_client.get_global_model2()
                # server.get_global_model2()
                FL.center_client.despatch(num=args.pos_num, round=FL.round)
                FL.round += 1
                FL.update_center()

                for i in range(args_.wick_num):
                    FL.clients[20 + i] = wick_Client(cid=int(i) + args.num_client, args=args, cname=FL.name)

                torch.cuda.empty_cache()

                print('-------------------------------\n')

    else:
        for FL in Fed_Links:
            FL.round = 1
            FL.center_client.despatch(num=args.pos_num, round=FL.round)

    '''server.Collent_feat(feat_mode='train', num=20)
    server.mul_train(Epoch=10)
    server.Collent_feat(feat_mode='test', num=20)
    # server.mul_test()'''
    for FL in Fed_Links:
        FL.Collent_feat(feat_mode='train')
        FL.Collent_feat(feat_mode='test')
        FL.mul_test()

    server = Server(Fed_Links, args)
    server.Collent_feat(feat_mode='train')
    server.mul_train(Epoch=10)
    server.Collent_feat(feat_mode='test')
    server.mul_test()


if __name__ == "__main__":
    main(args_)
