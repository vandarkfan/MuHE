import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from functools import partial
from typing import Optional, Union, Iterable
from scipy.stats import chi

from ATT_module import ATT3DLayer
from torch.autograd import Variable

class BaseClass(torch.nn.Module):
    def __init__(self):
        super(BaseClass, self).__init__()
        self.cur_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_mrr = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)
        self.best_itr = torch.nn.Parameter(torch.tensor(0, dtype=torch.int32), requires_grad=False)
        self.best_hit1 = torch.nn.Parameter(torch.tensor(0, dtype=torch.float64), requires_grad=False)

class MyLoss(torch.nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.loss_fn = nn.CrossEntropyLoss(label_smoothing=0)
        return

    def forward(self, pred1, tar1):
        loss = self.loss_fn(pred1, tar1)
        return loss


class MSLE(torch.nn.Module):
    def __init__(self, num_rel, embedding_dim=300, input_drop=0.4, feature_map_drop=0.3,
                 k_w=8, k_h=8, k_c=8, output_channel=8, array=2):
        super(MSLE, self).__init__()
        # 定义模型
        self.array = array
        self.att = ATT3DLayer(output_channel, reduction=int(output_channel*2))
        self.embedding_dim = embedding_dim
        self.perm = 1
        self.k_w = k_w
        self.k_h = k_h
        self.k_c = k_c
        self.device = torch.device('cuda')

        # 定义尺寸
        self.chequer_perm = self.get_chequer_perm()
        self.reshape_H = k_h
        self.reshape_W = k_w
        self.reshape_C = k_c
        self.in_channel = array  # 输入通道数
        self.out_1 = output_channel  # 第一个卷积核的输出通道数
        self.out_2 = output_channel  # 第二个卷积核的输出通道数
        self.out_3 = output_channel  # 第三个卷积核的输出通道数
        # 卷积核

        self.filter1_size = (1,1,array)
        self.filter2_size = (1,array,array)
        self.filter3_size = (array,array,array)
        self.h1 = self.filter1_size[0]
        self.w1 = self.filter1_size[1]
        self.c1 = self.filter1_size[2]
        self.h2 = self.filter2_size[0]
        self.w2 = self.filter2_size[1]
        self.c2 = self.filter2_size[2]
        self.h3 = self.filter3_size[0]
        self.w3 = self.filter3_size[1]
        self.c3 = self.filter3_size[2]
        filter1_dim = self.in_channel * self.out_1 * self.h1 * self.w1 * self.c1 
        self.filter1 = torch.nn.Embedding(num_rel, filter1_dim, padding_idx=0) 
        filter2_dim = self.in_channel * self.out_2 * self.h2 * self.w2 * self.c2 
        self.filter3 = torch.nn.Embedding(num_rel, filter2_dim, padding_idx=0)  
        filter3_dim = self.in_channel * self.out_3 * self.h3 * self.w3 * self.c3  
        self.filter5 = torch.nn.Embedding(num_rel, filter3_dim, padding_idx=0)  
        self.input_drop = torch.nn.Dropout(input_drop)
        self.feature_map_drop = torch.nn.Dropout3d(feature_map_drop)
        self.bn0 = torch.nn.BatchNorm3d(self.in_channel)
        self.bn1 = torch.nn.BatchNorm3d(self.out_1 + self.out_2 + self.out_3)
        self.bn1_1 = torch.nn.BatchNorm3d(self.out_1)
        self.bn1_2 = torch.nn.BatchNorm3d(self.out_2)
        self.bn1_3 = torch.nn.BatchNorm3d(self.out_3)
        self.bnall = torch.nn.BatchNorm3d(self.out_3 * 3)
        self.fc = nn.Linear(int(output_channel * output_channel * output_channel * self.array * output_channel * 3), self.embedding_dim)
        torch.nn.init.xavier_normal_(self.filter1.weight.data)
        torch.nn.init.xavier_normal_(self.filter3.weight.data)
        torch.nn.init.xavier_normal_(self.filter5.weight.data)
        nn.init.xavier_uniform_(self.fc.weight)
        self.fc.bias.data.zero_()
    def get_chequer_perm(self):
        ent_perm = np.int32([np.random.permutation(self.embedding_dim * (self.array-1)) for _ in range(self.perm)])  # 返回一个随机排列
        rel_perm = np.int32([np.random.permutation(self.embedding_dim) for _ in range(self.perm)])
        comb_idx = []
        temp = []
        ent_idx, rel_idx = 0, 0
        for i in range(self.k_c):
            for j in range(self.k_h):
                for k in range(self.k_w):
                    for m in range(self.array):
                        if (i + j) % self.array == m:
                            temp.append(rel_perm[0, rel_idx] + self.embedding_dim * (self.array - 1))
                            rel_idx += 1
                        else:
                            temp.append(ent_perm[0, ent_idx])
                            ent_idx += 1

        comb_idx.append(temp)

        chequer_perm = torch.LongTensor(np.int32(comb_idx)).to(self.device)
        return chequer_perm

    def forward(self, concat_input, rel):

        chequer_perm = concat_input[:, self.chequer_perm]
        stack_inp = chequer_perm.reshape((-1, self.array, self.k_w, self.k_h, self.k_c))
        x = self.bn0(stack_inp)
        x = self.input_drop(x)
        x = x.permute(1, 0, 2, 3, 4)

        f1 = self.filter1(rel)
        f1 = f1.reshape(concat_input.size(0) * self.out_1, self.in_channel, self.h1, self.w1, self.c1)
        f3 = self.filter3(rel)
        f3 = f3.reshape(concat_input.size(0) * self.out_2, self.in_channel, self.h2, self.w2, self.c2)
        f5 = self.filter5(rel)
        f5 = f5.reshape(concat_input.size(0) * self.out_3, self.in_channel, self.h3, self.w3, self.c3)


        x1 = F.conv3d(x, f1, groups=concat_input.size(0),padding='same')
        x1 = x1.reshape(concat_input.size(0), self.out_1, self.reshape_H, self.reshape_W, self.k_c)
        x1 = self.bn1_1(x1)


        x3 = F.conv3d(x, f3, groups=concat_input.size(0),padding='same')
        x3 = x3.reshape(concat_input.size(0), self.out_2, self.reshape_H, self.reshape_W, self.k_c)
        x3 = self.bn1_2(x3)


        x5 = F.conv3d(x, f5, groups=concat_input.size(0),padding='same')
        x5 = x5.reshape(concat_input.size(0), self.out_3, self.reshape_H, self.reshape_W, self.k_c)
        x5 = self.bn1_3(x5)

        x = x1 + x3 + x5
        y1, y3, y5 = self.att(x)
        y1 = y1.expand_as(x1)
        y3 = y3.expand_as(x3)
        y5 = y5.expand_as(x5)
        x1 = x1 * y1
        x3 = x3 * y3
        x5 = x5 * y5
        x = torch.cat([x1, x3, x5], dim=1)

        x = torch.relu(x)
        x = self.feature_map_drop(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x



class MuHE3D(BaseClass):

    def __init__(self, n_ent, n_rel, emb_dim, max_arity, device, ent2vis, ent2txt,
                 emb_dropout, vis_dropout, txt_dropout,
                 input_drop_2, feature_map_drop_2,
                 input_drop_3, feature_map_drop_3,
                 input_drop_4, feature_map_drop_4,
                 input_drop_5, feature_map_drop_5,
                 input_drop_6, feature_map_drop_6,
                 input_drop_7, feature_map_drop_7,
                 input_drop_8, feature_map_drop_8,
                 input_drop_9, feature_map_drop_9):
        super(MuHE3D, self).__init__()
        self.loss = MyLoss()
        self.n_ent = n_ent
        self.save_rel = n_rel
        self.n_rel = n_rel * max_arity
        self.device = device
        self.emb_dim = emb_dim

        self.lmbda = 0.05
        self.max_arity = max_arity
        self.ent_embeddings = nn.Parameter(torch.Tensor(self.n_ent, self.emb_dim))
        self.rel_embeddings = nn.Parameter(torch.Tensor(self.n_rel, self.emb_dim))
        self.pos_embeddings = nn.Embedding(self.max_arity, self.emb_dim)
        self.lp_token = nn.Parameter(torch.Tensor(1, self.emb_dim))
        self.ent_vis = ent2vis
        self.ent_txt = ent2txt

        self.str_ent_ln = nn.LayerNorm(self.emb_dim)
        self.str_rel_ln = nn.LayerNorm(self.emb_dim)
        self.vis_ln = nn.LayerNorm(self.emb_dim)
        self.txt_ln = nn.LayerNorm(self.emb_dim)
        self.all_ln = nn.LayerNorm(self.emb_dim)


        self.embdr = nn.Dropout(p=emb_dropout)
        self.visdr = nn.Dropout(p=vis_dropout)
        self.txtdr = nn.Dropout(p=txt_dropout)

        self.pos_str_ent = nn.Parameter(torch.Tensor(1, 1, self.emb_dim))
        self.pos_vis_ent = nn.Parameter(torch.Tensor(1, 1, self.emb_dim))
        self.pos_txt_ent = nn.Parameter(torch.Tensor(1, 1, self.emb_dim))

        self.proj_ent_vis = nn.Linear(self.ent_vis.shape[-1], self.emb_dim)
        self.proj_txt = nn.Linear(self.ent_txt.shape[-1], self.emb_dim)

        self.bn1 = nn.BatchNorm3d(num_features=1)
        self.pool = torch.nn.MaxPool3d((4, 1, 1))
        self.pool2d = torch.nn.MaxPool2d((1, 2))
        h_c = round(pow(self.emb_dim, 1/3))
        self.h_c = h_c
        self.dropout = nn.Dropout(0.4)
        self.conv_size_2d = (self.emb_dim) * 16 // 2
        self.MSLE_decoder_2 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=2, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_2, feature_map_drop=feature_map_drop_2)
        self.MSLE_decoder_3 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=3, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_3, feature_map_drop=feature_map_drop_3)
        self.MSLE_decoder_4 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=4, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_4, feature_map_drop=feature_map_drop_4)
        self.MSLE_decoder_5 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=5, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_5, feature_map_drop=feature_map_drop_5)
        self.MSLE_decoder_6 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=6, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_6, feature_map_drop=feature_map_drop_6)
        self.MSLE_decoder_7 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=7, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_7, feature_map_drop=feature_map_drop_7)
        self.MSLE_decoder_8 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=8, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_8, feature_map_drop=feature_map_drop_8)
        self.MSLE_decoder_9 = MSLE(num_rel = self.n_rel, embedding_dim=self.emb_dim, array=9, k_w=h_c, k_h=h_c, k_c=h_c, output_channel=h_c, input_drop=input_drop_9, feature_map_drop=feature_map_drop_9)

        self.nonlinear = nn.ReLU()
        self.conv_size = (self.emb_dim1 * self.emb_dim2) * 8 // 4
        self.register_parameter('b', nn.Parameter(torch.zeros(n_ent)))
        self.criterion = nn.Softplus()
        nn.init.xavier_uniform_(self.rel_embeddings.data)
        nn.init.xavier_uniform_(self.ent_embeddings.data)
        nn.init.xavier_uniform_(self.pos_embeddings.weight.data)
        nn.init.xavier_uniform_(self.proj_ent_vis.weight)
        nn.init.xavier_uniform_(self.proj_txt.weight)
        nn.init.xavier_uniform_(self.lp_token)
        nn.init.xavier_uniform_(self.pos_str_ent)
        nn.init.xavier_uniform_(self.pos_vis_ent)
        nn.init.xavier_uniform_(self.pos_txt_ent)
        self.proj_ent_vis.bias.data.zero_()
        self.proj_txt.bias.data.zero_()

    def get_param(self, shape):
        param = nn.Parameter(torch.Tensor(*shape))
        nn.init.xavier_normal_(param, gain=nn.init.calculate_gain('relu'))
        return param
    def getent_embeddings(self):
        rep_ent_str = self.embdr(self.str_ent_ln(self.ent_embeddings)).unsqueeze(1) + self.pos_str_ent
        rep_ent_vis = self.visdr(
            self.vis_ln(self.proj_ent_vis(self.ent_vis))) + self.pos_vis_ent  # 41105,1,768 -> 41105,1,256
        rep_ent_txt = self.txtdr(self.txt_ln(self.proj_txt(self.ent_txt))) + self.pos_txt_ent
        # sum
        ent_seq = torch.cat([rep_ent_str, rep_ent_vis, rep_ent_txt], dim=1) # 41105,4,256
        ent_embs = self.all_ln(torch.sum(ent_seq, dim=1).squeeze(1))

        return ent_embs

    def conv3d_process(self, concat_input, rid):
        array = concat_input.shape[1]
        concat_input = concat_input.view(concat_input.shape[0], -1)
        if array == 2:
            x = self.MSLE_decoder_2(concat_input,rid)

        if array == 3:
            x = self.MSLE_decoder_3(concat_input,rid)

        if array == 4:
            x = self.MSLE_decoder_4(concat_input,rid)

        if array == 5:
            x = self.MSLE_decoder_5(concat_input,rid)

        if array == 6:
            x = self.MSLE_decoder_6(concat_input,rid)

        if array == 7:
            x = self.MSLE_decoder_7(concat_input, rid)

        if array == 8:
            x = self.MSLE_decoder_8(concat_input, rid)
        if array == 9:
            x = self.MSLE_decoder_9(concat_input, rid)
        return x


    def forward(self, rel_idx, ent_idx, miss_ent_domain, ent_emb):
        rel_idx = rel_idx + (miss_ent_domain-1)*self.save_rel
        r = self.rel_embeddings[rel_idx].unsqueeze(1)
        ents = ent_emb[ent_idx]

        pos = [i for i in range(ent_idx.shape[1] + 1) if i + 1 != miss_ent_domain]
        pos = torch.tensor(pos).to(self.device)
        pos = pos.unsqueeze(0).expand_as(ent_idx)
        ents = ents + self.pos_embeddings(pos)

        concat_input = torch.cat((r, ents), dim=1)
        x = self.conv3d_process(concat_input, rel_idx)

        miss_ent_domain = torch.LongTensor([miss_ent_domain - 1]).to(self.device)
        mis_pos = self.pos_embeddings(miss_ent_domain)
        tar_emb = ent_emb + mis_pos

        scores = torch.mm(x, tar_emb.transpose(0, 1))
        scores += self.b.expand_as(scores)

        return scores
