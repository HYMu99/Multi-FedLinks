import os
import os.path as osp
import torchvision.datasets as datasets
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import shutil

class All_Client_Dataset(object):
    def __init__(self,root_dir, split_dir, args, re_divide = True,ex_pretrained_root = None, c_name = 'FV'):
        self.args = args
        self.c_name = c_name
        self.pretrained_num = args.pretrained_num
        self.pre_dir = os.path.join(split_dir, 'pre_train')
        self.pre_dir = os.path.join(self.pre_dir, c_name)
        if not os.path.exists(self.pre_dir):
            os.mkdir(self.pre_dir)
        elif re_divide:
            shutil.rmtree(self.pre_dir)
            os.mkdir(self.pre_dir)

        self.datasize_list = args.didata_list
        self.data_name = args.data_name
        self.root_dir = root_dir
        self.num_client = args.num_client
        self.batch_size = args.batch_size
        self.ex_pretrained_root = ex_pretrained_root
        self.split_dir = split_dir
        self.re_divide = re_divide
        self.dataset_dir = osp.join(self.split_dir, 'split_train_%s_c%04d' % (self.data_name, self.num_client))

        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        self.dataset_dir = osp.join(self.dataset_dir, self.c_name)
        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)
        elif self.re_divide == True:
            shutil.rmtree(self.dataset_dir)
            os.mkdir(self.dataset_dir)
        self.transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ])
        ###
        self.test_transform = transforms.Compose(
            [transforms.Resize((224, 224)),
             transforms.ToTensor(),
             transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
             ])
        # create train dataset for each client
        self.creating_each_client()
        # create infer dataset for each client
        # self.creating_infer_each_client()
        self.creating_public_dataset()

    def divide_file(self, size=None):
        or_data_train = self.root_dir + '/train'
        or_data_val = self.root_dir + '/val'
        or_data_test = self.root_dir + '/test'
        train_list = os.listdir(or_data_train)
        val_list = os.listdir(or_data_val)
        test_list = os.listdir(or_data_test)

        if self.pretrained_num != 0:
            if self.ex_pretrained_root:
                por_data_train = self.ex_pretrained_root + '/train'
                por_data_test = self.ex_pretrained_root + '/test'
                ptrain_list = os.listdir(por_data_train)
                pre_train_list = ptrain_list[:self.pretrained_num]
            else:
                por_data_train = or_data_train
                por_data_test = or_data_test
                pre_train_list = train_list[:self.pretrained_num]
            pre_data_train = self.pre_dir + '/train'
            pre_data_test = self.pre_dir + '/test'
            if not os.path.exists(pre_data_train):
                os.mkdir(pre_data_train)
                os.mkdir(pre_data_test)

            for i, id in enumerate(pre_train_list):
                oldf_train_id = por_data_train + '/' + str(id)
                newf_train_id = pre_data_train + '/' + str(id)
                oldf_test_id = por_data_test + '/' + str(id)
                newf_test_id = pre_data_test + '/' + str(id)
                shutil.copytree(oldf_train_id, newf_train_id)
                shutil.copytree(oldf_test_id, newf_test_id)

        if not self.ex_pretrained_root:
            train_list = train_list[self.pretrained_num:]
            val_list = val_list[self.pretrained_num:]
            test_list = test_list[self.pretrained_num:]

        if size == None:
            class_number = len(train_list)
            each_client_number = class_number // self.num_client
            order = np.arange(class_number)
            # np.random.shuffle(order)
            order = order.tolist()
            client_id = 0
            k = 0
            for i, id in enumerate(order):
                client_dir = osp.join(self.dataset_dir, 'client_%04d' % (client_id))
                ne_data_train = client_dir + '/train'
                ne_data_val = client_dir + '/val'
                ne_data_test = client_dir + '/test'
                if not os.path.exists(client_dir):
                    os.mkdir(client_dir)
                    os.mkdir(ne_data_train)
                    os.mkdir(ne_data_val)
                    os.mkdir(ne_data_test)

                assert train_list[id] == test_list[id]

                oldf_train_id = or_data_train + '/' + str(train_list[id])
                newf_train_id = ne_data_train + '/' + str(train_list[id])
                oldf_val_id = or_data_val + '/' + str(val_list[id])
                newf_val_id = ne_data_val + '/' + str(val_list[id])
                oldf_test_id = or_data_test + '/' + str(test_list[id])
                newf_test_id = ne_data_test + '/' + str(test_list[id])
                shutil.copytree(oldf_train_id, newf_train_id)
                if k < 10:
                    shutil.copytree(oldf_val_id, newf_val_id)
                    k += 1
                shutil.copytree(oldf_test_id, newf_test_id)

                if i + 1 >= (
                        (client_id + 1) * each_client_number) and i + 1 + each_client_number <= class_number:
                    client_id += 1
                    k = 0
        else:
            client_id = 0
            class_number = len(train_list)
            assert class_number == sum(size)

            order = np.arange(class_number)
            order = order.tolist()
            divi_num = size[client_id] - 1
            k = 0
            for i, id in enumerate(order):
                client_dir = osp.join(self.dataset_dir, 'client_%04d' % (client_id))
                ne_data_train = client_dir + '/train'
                ne_data_val = client_dir + '/val'
                ne_data_test = client_dir + '/test'
                if not os.path.exists(client_dir):
                    os.mkdir(client_dir)
                    os.mkdir(ne_data_train)
                    os.mkdir(ne_data_val)
                    os.mkdir(ne_data_test)

                assert train_list[id] == test_list[id]
                assert val_list[id] == test_list[id]

                oldf_train_id = or_data_train + '/' + str(train_list[id])
                newf_train_id = ne_data_train + '/' + str(train_list[id])
                oldf_val_id = or_data_val + '/' + str(val_list[id])
                newf_val_id = ne_data_val + '/' + str(val_list[id])
                oldf_test_id = or_data_test + '/' + str(test_list[id])
                newf_test_id = ne_data_test + '/' + str(test_list[id])
                shutil.copytree(oldf_train_id, newf_train_id)
                if k < 10:
                    shutil.copytree(oldf_val_id, newf_val_id)
                    k += 1
                shutil.copytree(oldf_test_id, newf_test_id)

                if i == divi_num and i != (class_number - 1):
                    client_id += 1
                    k = 0
                    divi_num = divi_num + size[client_id]

        print('Finish divide' + self.c_name + ' database')

    def creating_each_client(self):
        self.train_loaders = []
        self.train_dataset_sizes = []
        self.val_loaders = []
        self.val_dataset_sizes = []
        self.train_class_sizes = []
        self.test_loaders = []
        if self.re_divide:
            self.divide_file(self.datasize_list)

        for c in range(self.num_client):
            client_dir = os.path.join(self.dataset_dir,'client_%04d'%(c))
            traindir = osp.join(client_dir ,'train')
            valdir =osp.join(client_dir,'val')
            testdir = osp.join(client_dir , 'test')
            dataset = datasets.ImageFolder(traindir, self.transform)
            dataset_val = datasets.ImageFolder(valdir, self.transform)
            dataset_test = datasets.ImageFolder(testdir, self.transform)
            train_loader = torch.utils.data.DataLoader( dataset,
                batch_size=self.args.batch_size, shuffle=True, pin_memory=True)
            val_loader = torch.utils.data.DataLoader( dataset_val,
                                                       batch_size=self.args.batch_size, shuffle=False, pin_memory=True)
            test_loader = torch.utils.data.DataLoader( dataset_test,
                                                       batch_size=self.args.batch_size, shuffle=False, pin_memory=True)

            self.train_loaders.append(train_loader)
            self.test_loaders.append(test_loader)
            self.val_loaders.append(val_loader)
            self.train_dataset_sizes.append(len(dataset))
            self.train_class_sizes.append(len(dataset)//3)

    def creating_public_dataset(self):
        public_train_dir = self.pre_dir + '/train'
        public_test_dir = self.pre_dir + '/test'
        dataset_train = datasets.ImageFolder(public_train_dir, self.transform)
        dataset_test = datasets.ImageFolder(public_test_dir, self.transform)
        self.public_train_loader = DataLoader(dataset=dataset_train,batch_size=self.args.batch_size,shuffle=True,pin_memory=True)
        self.public_test_loader = DataLoader(dataset=dataset_test,batch_size=self.args.batch_size,shuffle=False,pin_memory=True)
