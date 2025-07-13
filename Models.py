import torch
import torch.nn as nn
import torch.nn.functional as F
# import pretrainedmodels
import os


class Block(nn.Module):
    def __init__(self, in_channel, out_channel, strides=1, same_shape=True):
        super(Block, self).__init__()
        self.same_shape = same_shape
        if not same_shape:
            strides = 2
        self.strides = strides
        self.block = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=strides, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel)
        )
        if not same_shape:
            self.conv3 = nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=strides, bias=False)
            self.bn3 = nn.BatchNorm2d(out_channel)

    def forward(self, x):
        out = self.block(x)
        if not self.same_shape:
            x = self.bn3(self.conv3(x))
        return F.relu(out + x)


class MyNet(nn.Module):
    def __init__(self, num_classes=512):
        super().__init__()
        # 最开始的几层
        self.pre = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2, 1))
        self.layer1 = self._make_layer(32, 32, 1)
        self.layer2 = self._make_layer(32, 64, 1, same_shape=False)
        self.layer3 = self._make_layer(64, 128, 2, same_shape=False)
        self.layer4 = self._make_layer(128, 512, 1, same_shape=False)

        # 分类用的全连接
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, in_channel, out_channel, block_num, stride=1, same_shape=True):
        layers = []
        if stride != 1:
            layers.append(Block(in_channel, out_channel, stride, same_shape=False))
        else:
            layers.append(Block(in_channel, out_channel, stride, same_shape=same_shape))

        for i in range(1, block_num):
            layers.append(Block(out_channel, out_channel))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.pre(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = F.avg_pool2d(x, 7)
        x = x.view(x.size(0), -1)
        return self.fc(x)


class Model():
    def __init__(self, name, class_number, model_path, is_FC=True, is_wicked=False):
        super().__init__()
        self.name = name
        '''self.model = pretrainedmodels.__dict__[self.name]()
        num_fc_ftr = self.model.last_linear.in_features
        self.model.last_linear = torch.nn.Linear(num_fc_ftr, 512)'''
        self.model = MyNet()
        if is_FC:
            self.fc = torch.nn.Linear(512, class_number)
        self.model_path = model_path
        if is_wicked:
            if os.path.exists('./c_pth/wick_model.pth.tar'):
                checkpoint = torch.load('./c_pth/wick_model.pth.tar')
                self.model.load_state_dict(checkpoint['state_dict'])
            else:
                torch.save({'state_dict': self.model.state_dict()}, './c_pth/wick_model.pth.tar')

    def load_model(self):
        checkpoint = torch.load(self.model_path + '/local_model.pth.tar')
        self.model.load_state_dict(checkpoint['state_dict'])
        if os.path.exists(self.model_path + '/local_fc.pth.tar'):
            checkpoint = torch.load(self.model_path + '/local_fc.pth.tar')
            self.fc.load_state_dict(checkpoint['state_dict'])
        return self.model, self.fc

    def save_model(self):
        torch.save({'state_dict': self.model.state_dict()}, self.model_path + 'local_model.pth.tar')
        torch.save({'state_dict': self.fc.state_dict()}, self.model_path + 'local_fc.pth.tar')


class Muti_Model():
    def __init__(self):
        self.mul_fc = torch.nn.Linear(1024, 512)
        self.test_data_loder = []
        self.train_data_loder = []
        self.print_dic = {}
        self.vein_dic = {}
        self.print_dic2 = {}
        self.vein_dic2 = {}
        self.co_lables = []
        self.feat_data = 'test'

    def co_label(self, mode):
        self.feat_data = mode
        self.co_lables = []
        if self.feat_data == 'test':
            self.print_dic = dict(sorted(self.print_dic.items(), key=lambda x: x[0]))
            self.vein_dic = dict(sorted(self.vein_dic.items(), key=lambda x: x[0]))
            for key in self.print_dic.keys():
                if key in self.vein_dic.keys():
                    if key not in self.co_lables:
                        self.co_lables.append(key)
                    for i in range(min(len(self.print_dic[key]), len(self.vein_dic[key]))):
                        data = torch.cat([self.print_dic[key][i], self.vein_dic[key][i]], -1)
                        self.test_data_loder.append([key, data])
            print('Finish test co_data')
        else:
            self.print_dic2 = dict(sorted(self.print_dic2.items(), key=lambda x: x[0]))
            self.vein_dic2 = dict(sorted(self.vein_dic2.items(), key=lambda x: x[0]))
            for key in self.print_dic2.keys():
                if key in self.vein_dic2.keys():
                    if key not in self.co_lables:
                        self.co_lables.append(key)
                    for i in range(min(len(self.print_dic2[key]), len(self.vein_dic2[key]))):
                        data = torch.cat([self.print_dic2[key][i], self.vein_dic2[key][i]], -1)
                        self.train_data_loder.append([key, data])
            print('Finish train co_data')

            self.sum_class_number = len(self.co_lables)
            self.fc = torch.nn.Linear(512, self.sum_class_number)

    def collen_test_data(self, client):
        self.feat_data = 'test'
        for key, val in client.test_feat.items():
            if client.c_name == 'FP':
                self.print_dic[key] = val
            elif client.c_name == 'FV':
                self.vein_dic[key] = val
            else:
                print('No c_name in server')

    def collen_train_data(self, client):
        self.feat_data = 'train'
        for key, val in client.train_feat.items():
            if client.c_name == 'FP':
                self.print_dic2[key] = val
            elif client.c_name == 'FV':
                self.vein_dic2[key] = val
            else:
                print('No c_name in server')
