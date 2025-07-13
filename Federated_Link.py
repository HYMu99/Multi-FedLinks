import os
from Models import Muti_Model
import numpy as np
from tqdm import tqdm

class Federated_Link(object):
    def __init__(self, clients, name, client):
        self.clients = clients
        self.name = name
        self.center_client = client
        self.round = -1
        self.Connect_client()

        self.linear = Muti_Model()
        self.mul_feat = {}
        self.FAR_cosdis = []
        self.FRR_cosdis = []

    def Connect_client(self):
        for client in self.clients:
            client.Fed_Link(self.clients)

    def update_center(self):
        self.center_client = self.clients[self.center_client.next_center]

    def Collent_feat(self,feat_mode = 'train',num = 20):
        for client in self.clients[:num]:
            client.test(is_fate = True,feat_data = feat_mode)
            if feat_mode == 'train':
                self.linear.collen_train_data(client)
            elif feat_mode == 'test':
                self.linear.collen_test_data(client)

        if self.name == 'FP':
            for lable,feats in self.linear.print_dic.items():
                lable = int(lable)
                feat_list = feats
                for i, feat in enumerate(feat_list):
                    feat = [item.cpu().detach().numpy() for item in feat]
                    if lable not in self.mul_feat.keys():
                        self.mul_feat[lable] = [feat]
                    else:
                        self.mul_feat[lable].append(feat)
        elif self.name == 'FV':
            for lable,feats in self.linear.vein_dic.items():
                lable = int(lable)
                feat_list = feats
                for i, feat in enumerate(feat_list):
                    feat = [item.cpu().detach().numpy() for item in feat]
                    if lable not in self.mul_feat.keys():
                        self.mul_feat[lable] = [feat]
                    else:
                        self.mul_feat[lable].append(feat)

    def Feat_Data(self):
        if self.linear.feat_data == 'train':
            label_list = [(int(data[0])-1) for data in self.linear.train_data_loder]
            data_list = [data[1] for data in self.linear.train_data_loder]
        else:
            label_list = [(int(data[0]) - 1) for data in self.linear.test_data_loder]
            data_list = [data[1] for data in self.linear.test_data_loder]
        return label_list,data_list

    def mul_test(self):
        self.test_feat_data_load()
        self.cul_cosdistence()

    def test_feat_data_load(self):
        if not os.path.exists('./pair'):
            os.mkdir('./pair')
        self.FAR_feat_data = []
        self.FRR_feat_data = []

        for label, feats in tqdm(self.mul_feat.items(),desc='load_data'):
            k = 0
            for i, feat in enumerate(feats):
                j = i + 1
                while j < len(feats):
                    self.FRR_feat_data.append([feat,feats[j]])
                    j += 1

                for ot_label, ot_feats in self.mul_feat.items():
                    if label >= ot_label:
                        continue
                    # rod = random.randint(0,len(ot_feats)-1)
                    for ot_feat in ot_feats:
                        self.FAR_feat_data.append([feat, ot_feat])


    def cul_cosdistence(self):
        for far in tqdm(self.FAR_feat_data,desc=self.name + ' Cul_FAR_cos'):
            cos = np.dot(far[0], far[1]) / (np.linalg.norm(far[0]) * np.linalg.norm(far[1]))
            self.FAR_cosdis.append(cos)
        np.save('./pair' + '/FAR_cos' + self.name, self.FAR_cosdis)
        for frr in tqdm(self.FRR_feat_data,desc=self.name + ' Cul_FAR_cos'):
            cos = np.dot(frr[0], frr[1]) / (np.linalg.norm(frr[0]) * np.linalg.norm(frr[1]))
            self.FRR_cosdis.append(cos)
        np.save('./pair' + '/FRR_cos' + self.name, self.FRR_cosdis)
