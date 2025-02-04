import copy
import math
import torch
import argparse
import numpy as np
import torch.nn as nn
from math import sqrt
import torch.nn.functional as F
from models.resnet3d import resnet18,resnet34,resnet50


class MoE(nn.Module):
    def __init__(self, input_dim, num_experts, args):
        super(MoE, self).__init__()
        self.num_experts = num_experts
        self.args = args
        self.gating_type = self.args.gating_type
        self.moe_fusion_type = self.args.moe_fusion_type

        self.experts = nn.ModuleList()
        for _ in range(num_experts):
            self.experts.append(nn.Sequential(nn.ReLU(), nn.Linear(input_dim, input_dim)))

        if self.gating_type == 'concat':
            if self.args.gating_concat_mlp_layer =='multi':
                # # 2023-11-30-17_45
                self.gating_fc = nn.Sequential(nn.ReLU(),
                                               nn.Linear(input_dim * num_experts, input_dim),
                                               nn.ReLU(),
                                               nn.Linear(input_dim, num_experts))
            else:
                self.gating_fc = nn.Sequential(nn.ReLU(),
                                               nn.Linear(input_dim * num_experts, num_experts))
        elif self.gating_type == 'importance':
            self.gating_fc = nn.Sequential(nn.ReLU(),
                                           nn.Linear(input_dim * num_experts, input_dim))

        self.flatten = nn.Flatten(start_dim=1)
        self.softmax = nn.Softmax(dim=1)
        self.sigmoid = nn.Sigmoid()

    def gating_network(self, x):
        if self.gating_type == 'concat':
            weights = self.softmax(self.gating_fc(self.flatten(x))) # [bs, num_experts]
        elif self.gating_type == 'importance':
            sample_fusion = torch.unsqueeze(self.gating_fc(self.flatten(x)), -1)  # [bs, input_dim, 1]
            weights = self.softmax(torch.squeeze(torch.bmm(x, sample_fusion), dim=-1))  # [bs, num_experts]
        elif self.gating_type == 'correlation':
            input = self.sigmoid(x)
            corrs = torch.bmm(input,torch.transpose(input,-1,-2)) # [bs, num_experts, num_experts]
            weights = self.softmax(torch.sum(corrs, dim=-1))
        return weights

    def forward(self, x):
        gating_input = torch.stack([x[i] for i in range(self.num_experts)], dim=1) # [bs, num_experts, input_dim]
        gating_weights = self.gating_network(gating_input) # [bs, num_experts]

        expert_outputs = torch.stack([self.experts[i](x[i]) for i in range(self.num_experts)], dim=1)  # [bs, num_experts, input_dim]

        if self.moe_fusion_type == 'sum':
            moe_output = torch.sum(expert_outputs * torch.unsqueeze(gating_weights, dim=-1), dim=1)  # [bs, input_dim]
        elif self.moe_fusion_type == 'concat':
            moe_output = self.flatten(expert_outputs * torch.unsqueeze(gating_weights, dim=-1))  # [bs, input_dim * num_experts]

        return {'moe_output': moe_output,
                'gating_weights': gating_weights}


class CFDL(torch.nn.Module):

    def __init__(self, args=None, criterion=None, dim_list=None, num_classes=None):
        super(CFDL, self).__init__()

        self.feature_dim = args.feature_dim
        # self.feature_dim = 32
        self.args = args
        self.criterion = criterion
        self.num_classes = num_classes
        self.eps = torch.tensor(0.00000001, dtype=torch.float32, requires_grad=False)
        self.squared_l2norm = nn.MSELoss(reduction='sum')
        self.mse_loss = nn.MSELoss(reduction='mean')
        self.l1_loss = nn.L1Loss()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.if_allpair = self.args.if_allpair

        if self.args.dataset_name == 'men' or 'mrnet' in self.args.dataset_name:
            self.modal_num = 3
        elif self.args.dataset_name == 'brats':
            self.modal_num = 4

        if self.args.model_depth == 18 or self.args.model_depth == 34:
            self.dim_out = 512
        elif self.args.model_depth == 50:
            self.dim_out = 2048
        else:
            print('wrong model_depth!')

        if self.args.model_depth == 18:
            resnet = resnet18(modal_num=1)
        elif self.args.model_depth == 34:
            resnet = resnet34(modal_num=1)
        elif self.args.model_depth == 50:
            resnet = resnet50(modal_num=1)

        backbone = list(resnet.children())

        backbone_base = nn.Sequential(*backbone[:10])

        # backbone
        self.backbone_encoder = nn.ModuleList([])
        for i in range(self.modal_num):
            self.backbone_encoder.append(nn.Sequential(copy.deepcopy(backbone_base),
                                                       nn.Dropout(p=self.args.encoder_drop)))

        ### dis ###
        self.encoder_hidden = [256, self.feature_dim]

        # share
        self.linear_dis_share = nn.Sequential(nn.ReLU(), nn.Linear(self.dim_out, self.feature_dim))
        # spec
        self.linear_dis_spec = nn.ModuleList([])
        for i in range(self.modal_num):
            self.linear_dis_spec.append(nn.Sequential(nn.ReLU(), nn.Linear(self.dim_out, self.feature_dim)))
        # sup
        if self.args.ifdis_sup:
            self.linear_dis_sup = nn.ModuleList([])
            if self.modal_num == 3:
                self.dis_sup_num = 3
                # dis_num_0: modal_0 && modal_1
                # dis_num_1: modal_0 && modal_2
                # dis_num_2: modal_1 && modal_2
            elif self.modal_num == 4:
                if self.if_allpair:
                    self.dis_sup_num = 10
                    # two modals
                    # dis_num_0: modal_0 && modal_1
                    # dis_num_1: modal_0 && modal_2
                    # dis_num_2: modal_0 && modal_3
                    # dis_num_3: modal_1 && modal_2
                    # dis_num_4: modal_1 && modal_3
                    # dis_num_5: modal_2 && modal_3
                    # three modals
                    # dis_num_0: modal_0 && modal_1 && modal_2
                    # dis_num_1: modal_0 && modal_1 && modal_3
                    # dis_num_2: modal_0 && modal_2 && modal_3
                    # dis_num_3: modal_1 && modal_2 && modal_3
                else:
                    self.dis_sup_num = 8
                    # two modals
                    # dis_num_0: modal_0 && modal_1
                    # dis_num_1: modal_0 && modal_3
                    # dis_num_2: modal_1 && modal_2
                    # dis_num_3: modal_2 && modal_3
                    # three modals
                    # dis_num_0: modal_0 && modal_1 && modal_2
                    # dis_num_1: modal_0 && modal_1 && modal_3
                    # dis_num_2: modal_0 && modal_2 && modal_3
                    # dis_num_3: modal_1 && modal_2 && modal_3
            for i in range(self.dis_sup_num):
                self.linear_dis_sup.append(nn.Sequential(nn.ReLU(), nn.Linear(self.dim_out, self.feature_dim)))


        if self.args.ifdis_sup:
            feature_number = self.modal_num + self.dis_sup_num + 1
        else:
            feature_number = self.modal_num + 1

        if self.args.fusion_type == 'moe':
            self.moe_model = MoE(input_dim=self.feature_dim, num_experts=feature_number, args=args)

        # main cls
        if self.args.fusion_type == 'concat':
            self.linear_cls = nn.Sequential(nn.Linear(self.feature_dim*feature_number, self.feature_dim),
                                            nn.ReLU(),
                                            nn.Dropout(p=self.args.cls_drop),
                                            nn.Linear(self.feature_dim, self.num_classes))
        elif self.args.fusion_type == 'moe':
            if self.args.moe_fusion_type == 'sum':
                self.linear_cls = nn.Sequential(nn.Linear(self.feature_dim, 32),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.args.cls_drop),
                                                nn.Linear(32, self.num_classes))
            elif self.args.moe_fusion_type == 'concat':
                self.linear_cls = nn.Sequential(nn.Linear(self.feature_dim*feature_number, self.feature_dim),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.args.cls_drop),
                                                nn.Linear(self.feature_dim, 32),
                                                nn.ReLU(),
                                                nn.Dropout(p=self.args.cls_drop),
                                                nn.Linear(32, self.num_classes))
            if self.args.ifmoe_aux:
                self.linear_cls_aux = nn.ModuleList([])
                for i in range(feature_number):
                    self.linear_cls_aux.append(nn.Sequential(nn.Linear(self.feature_dim, 32),
                                                         nn.ReLU(),
                                                         nn.Dropout(p=self.args.cls_drop),
                                                         nn.Linear(32, self.num_classes)))

    def weight_expand(self, weight, shape):
        # new_label = label.repeat(1,shape[1]*shape[2]*shape[3]*shape[4])
        new_weight = weight.repeat(1, shape[1])
        new_weight = torch.reshape(new_weight, shape)
        return new_weight

    def mixup_data(self, x, y, alpha=1.0, use_cuda=True):
        '''Returns mixed inputs, pairs of targets, and lambda'''
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1

        batch_size = x.size()[0]
        if use_cuda:
            index = torch.randperm(batch_size).cuda()
        else:
            index = torch.randperm(batch_size)

        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        return mixed_x, y_a, y_b, lam

    def mixup_criterion(self, criterion, pred, y_a, y_b, lam):
        return lam * torch.mean(criterion(pred, y_a)) + (1 - lam) * torch.mean(criterion(pred, y_b))

    def calc_sim(self, x1, x2):
        if self.args.sim_loss_type == 'cosine':
            # return 1-torch.cosine_similarity(x1,x2,dim=-1)
            # return (torch.cosine_similarity(x1, x2, dim=-1)+1)/x1.shape[0] #求batch mean
            return torch.mean(torch.cosine_similarity(x1, x2, dim=-1)+1) #求batch mean
        elif self.args.sim_loss_type == 'l2norm':
            return self.mse_loss(x1,x2)
        elif self.args.sim_loss_type == 'MI':
            # x1_np = x1.cpu().numpy()
            # x2_np = x2.cpu().numpy()
            # mi_score = 0
            # for i in range(x1_np.shape[0]):
            #     mi_score = mi_score + mutual_info_score(x1_np[i], x2_np[i])
            # return torch.as_tensor(mi_score / x1_np.shape[0]).cuda()
            return self.mutual_info(self.sigmoid(x1),self.sigmoid(x2))

    def calc_con_loss(self, anchor, pos, neg_list):

        if self.args.sim_loss_type == 'cosine' or self.args.sim_loss_type == 'MI':
            cs_anchor_pos = torch.exp(self.calc_sim(anchor, pos) / self.args.tau)
            for i in range(len(neg_list)):
                if i == 0:
                    cs_anchor_negs = torch.exp(self.calc_sim(anchor, neg_list[i]) / self.args.tau)
                else:
                    cs_anchor_negs = cs_anchor_negs + torch.exp(self.calc_sim(anchor, neg_list[i]) / self.args.tau)
            loss = torch.mean(-torch.log(cs_anchor_pos / (cs_anchor_pos + cs_anchor_negs)+self.eps))

        elif self.args.sim_loss_type == 'l2norm':
            cs_anchor_pos = self.calc_sim(anchor, pos)
            for i in range(len(neg_list)):
                if i == 0:
                    cs_anchor_negs = self.calc_sim(anchor, neg_list[i])
                else:
                    cs_anchor_negs = cs_anchor_negs + self.calc_sim(anchor, neg_list[i])
            loss = (cs_anchor_pos + self.eps) / cs_anchor_negs
        else:
            print('wrong sim_loss_type input!')
        return loss

    def contrastive_loss(self, pos_list, neg_list):

        for i in range(len(pos_list)):
            anchor = pos_list[i]
            new_pos_list = list(range(len(pos_list)))
            new_pos_list.remove(i)
            for j in new_pos_list:
                pos = pos_list[j]
                if i == 0 and j == 1: # j从第一个开始
                    con_loss = self.calc_con_loss(anchor, pos, neg_list)
                else:
                    con_loss = con_loss + self.calc_con_loss(anchor, pos, neg_list)
        return con_loss

    def expand_dim(self, x, type):
        if type == '1d-3d':
            x = torch.unsqueeze(x, dim=-1)
            x = torch.unsqueeze(x, dim=-1)
            return x.expand(-1, 2, self.feature_dim).cuda()
        elif type == '1d-2d':
            x = torch.unsqueeze(x, dim=-1)
            return x.expand(-1, 2).cuda()
        elif type == '2d-3d':
            x = torch.unsqueeze(x, dim=-1)
            return x.expand(-1, -1, self.feature_dim).cuda()

    def sa_operation(self,x,name):
        norm_fact = 1 / sqrt(x.shape[1])
        if name == 'share':
            q = self.linear_q_share(x)
            k = self.linear_k_share(x)
            v = self.linear_v_share(x)
        elif name == 'k':
            q = self.linear_q_k(x)
            k = self.linear_k_k(x)
            v = self.linear_v_k(x)
        elif name == 'v':
            q = self.linear_q_v(x)
            k = self.linear_k_v(x)
            v = self.linear_v_v(x)
        elif name == 'q0':
            q = self.linear_q_q[0](x)
            k = self.linear_k_q[0](x)
            v = self.linear_v_q[0](x)
        elif name == 'q1':
            q = self.linear_q_q[1](x)
            k = self.linear_k_q[1](x)
            v = self.linear_v_q[1](x)
        elif name == 'q2':
            q = self.linear_q_q[2](x)
            k = self.linear_k_q[2](x)
            v = self.linear_v_q[2](x)
        elif name == 'q3':
            q = self.linear_q_q[3](x)
            k = self.linear_k_q[3](x)
            v = self.linear_v_q[3](x)
        else:
            print('wrong input name in fun(sa_operation)!')

        if self.args.ifsa_addrelu=='T':
            q = self.relu(q)
            k = self.relu(k)
            v = self.relu(v)

        att = torch.bmm(torch.unsqueeze(q, dim=2),torch.unsqueeze(k, dim=2).transpose(1, 2)) * norm_fact  # [batch,input_dim,input_dim]
        sa = torch.bmm(torch.softmax(att, dim=-1), torch.unsqueeze(v, dim=2))
        return torch.squeeze(sa,dim=-1)

    def cma_operation(self,q,k,v,norm_fact):
        att = torch.bmm(torch.unsqueeze(q, dim=2), torch.unsqueeze(k, dim=2).transpose(1, 2)) * norm_fact  # [batch,input_dim,input_dim]
        ca = torch.bmm(torch.softmax(att, dim=-1), torch.unsqueeze(v, dim=2))
        return ca

    def CMA(self, share, spe_list):
        x_fusion_list = None
        if self.args.cma_type == 'cascade_comp_inf': # cascade complete information
            x_fusion_list = []
            x_fusion_list.append(share)
            for i in range(len(spe_list)):
                if self.args.cascade_order == 'order':
                    j = i
                elif self.args.cascade_order == 'reverse':
                    j = len(spe_list)-1-i
                else:
                    print('Wrong cascade oreder input!')
                    break

                if i == 0:
                    if self.args.ifse_agg == 'T':
                        concat_fusion = torch.cat((torch.unsqueeze(share,dim=1),torch.unsqueeze(spe_list[j],dim=1)),dim=1)
                    x_fusion = self.cascade_comp_cma[j](share, spe_list[j], True)
                    x_fusion_list.append(x_fusion)
                else:
                    if self.args.ifse_agg == 'T':
                        concat_fusion = torch.cat((concat_fusion,torch.unsqueeze(spe_list[j],dim=1)),dim=1)
                    x_fusion = self.cascade_comp_cma[j](x_fusion, spe_list[j], False)
                    x_fusion_list.append(x_fusion)
            if self.args.ifse_agg == 'T':
                concat_fusion = torch.cat((concat_fusion, torch.unsqueeze(x_fusion, dim=1)),dim=1)
                x_se = self.se_agg_model(concat_fusion)
                out_att = x_se
            else:
                out_att = x_fusion
        elif self.args.cma_type == 'cascade_mutual_spe': # cascade complete information
            x_fusion_list = []
            x_fusion_list.append(share)
            for i in range(len(spe_list)):
                if self.args.cascade_order == 'order':
                    j = i
                elif self.args.cascade_order == 'reverse':
                    j = len(spe_list)-1-i
                else:
                    print('Wrong cascade oreder input!')
                    break

                if i == 0:
                    if self.args.ifse_agg == 'T':
                        concat_fusion = torch.cat((torch.unsqueeze(share,dim=1),torch.unsqueeze(spe_list[j],dim=1)),dim=1)
                    x_fusion = self.cascade_mutual_spe[j](share, spe_list[j], True)
                    x_fusion_list.append(x_fusion)
                else:
                    if self.args.ifse_agg == 'T':
                        concat_fusion = torch.cat((concat_fusion,torch.unsqueeze(spe_list[j],dim=1)),dim=1)
                    x_fusion = self.cascade_mutual_spe[j](x_fusion, spe_list[j], False)
                    x_fusion_list.append(x_fusion)
            if self.args.ifse_agg == 'T':
                concat_fusion = torch.cat((concat_fusion, torch.unsqueeze(x_fusion, dim=1)),dim=1)
                x_se = self.se_agg_model(concat_fusion)
                out_att = x_se
            else:
                out_att = x_fusion
        elif self.args.cma_type == 'comp_inf':
            out_att = self.cma_model(share, spe_list)
        elif self.args.cma_type == 'concat_q':
            # get sup information of v
            for i in range(len(spe_list)):
                if i == 0:
                    spe_concat = spe_list[i]
                else:
                    spe_concat = torch.cat((spe_concat,spe_list[i]),dim=-1)
            out_att = self.cascade_comp_cma(share, self.linear_spe(spe_concat), True)
        else:
            print('wrong cma_type input!')

        return {'out': out_att,
                'x_fusion_list':x_fusion_list}

    def dot_product_angle_np(self, v1, v2):
        v1 = np.squeeze(v1)
        v2 = np.squeeze(v2)
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0:
            print("Zero magnitude vector!")
        else:
            vector_dot_product = np.dot(v1, v2)
            arccos = np.arccos(vector_dot_product / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(arccos)
            return angle
        return 0

    def calc_cross_recon_loss(self,x):
        return torch.mean(torch.pow(torch.sum(x**2,dim=1),0.5))

    def orthogonality_operarion(self,input1, input2):

        batch_size = input1.size(0)
        input1 = input1.view(batch_size, -1)
        input2 = input2.view(batch_size, -1)

        # Zero mean
        input1_mean = torch.mean(input1, dim=0, keepdims=True)
        input2_mean = torch.mean(input2, dim=0, keepdims=True)
        input1 = input1 - input1_mean
        input2 = input2 - input2_mean

        input1_l2_norm = torch.norm(input1, p=2, dim=1, keepdim=True).detach()
        input1_l2 = input1.div(input1_l2_norm.expand_as(input1) + 1e-6)

        input2_l2_norm = torch.norm(input2, p=2, dim=1, keepdim=True).detach()
        input2_l2 = input2.div(input2_l2_norm.expand_as(input2) + 1e-6)

        sup_loss = torch.mean((input1_l2.t().mm(input2_l2)).pow(2))

        return sup_loss

    def calc_diff_loss(self, diff_list):
        for i in range(len(diff_list)):
            for j in range(i+1,len(diff_list)):
                if i == 0 and j == 1:
                    if self.args.diff_loss_type == 'l2norm':
                        loss = self.mse_loss(diff_list[i], diff_list[j])
                    elif self.args.diff_loss_type == 'orthogonality':
                        loss = self.orthogonality_operarion(diff_list[i], diff_list[j])
                    elif self.args.diff_loss_type == 'cosine':
                        loss = torch.mean(torch.abs(torch.cosine_similarity(diff_list[i], diff_list[j], dim=-1)))
                    else:
                        print('wrong diff_loss_type!')
                else:
                    if self.args.diff_loss_type == 'l2norm':
                        loss = loss + self.mse_loss(diff_list[i], diff_list[j])
                    elif self.args.diff_loss_type == 'orthogonality':
                        loss = loss + self.orthogonality_operarion(diff_list[i], diff_list[j])
                    elif self.args.diff_loss_type == 'cosine':
                        loss = loss + torch.mean(torch.abs(torch.cosine_similarity(diff_list[i], diff_list[j], dim=-1)))
                    else:
                        print('wrong diff_loss_type!')
        if self.args.diff_loss_type == 'l2norm':
            return 1/loss
        elif self.args.diff_loss_type == 'orthogonality' or self.args.diff_loss_type == 'cosine':
            return loss

    def cross_recon_op(self,x_ori_list,x_share_list,x_dif_list):
        # cross_recon_loss = 0
        for i in range(len(x_dif_list)):
            for j in range(len(x_share_list)):
                x_recon = self.decoder_recon[len(x_dif_list)*i+j](torch.cat((x_dif_list[i],x_share_list[j]),dim=-1))
                # cross_recon_loss = cross_recon_loss + self.calc_cross_recon_loss(x_ori_list[i]-x_recon)
                if i == 0 and j == 0:
                    cross_recon_loss = torch.sum(self.calc_sim(x_ori_list[i],x_recon)) #在calc_sim中已经求过mean了
                else:
                    cross_recon_loss = cross_recon_loss + torch.sum(self.calc_sim(x_ori_list[i],x_recon))
        return cross_recon_loss

    def recon_op(self, x_ori_list=None, x_share_list=None, x_share_sup_list=None, x_dif_list=None):
        if self.args.ifdis_sup:
            assert len(x_ori_list) == len(x_share_list) == len(x_share_sup_list) == len(x_dif_list)
        else:
            assert len(x_ori_list) == len(x_share_list) == len(x_dif_list)

        for i in range(len(x_ori_list)):
            if self.args.ifdis_sup:
                x_recon = self.decoder_recon[i](torch.cat((x_share_list[i], x_share_sup_list[i], x_dif_list[i]), dim=-1))
            else:
                x_recon = self.decoder_recon[i](torch.cat((x_share_list[i], x_dif_list[i]), dim=-1))

            if i == 0:
                recon_loss = self.mse_loss(x_ori_list[i], x_recon)
            else:
                recon_loss = recon_loss + self.mse_loss(x_ori_list[i], x_recon)
        return recon_loss/self.modal_num

    def one_hot_embedding(self,label, num_classes=2):
        # Convert to One Hot Encoding
        y = torch.eye(num_classes)
        return y[label]

    def sim_operation_between_3(self, x0, x1, x2):
        return self.calc_sim(x0, x1) + self.calc_sim(x0, x2) + self.calc_sim(x1, x2)

    def forward(self, x, label=None, global_step=None, run_type='train'):
        x_ori_list = []

        for i in range(self.modal_num):
            data = x[:, i, ...]
            if data.ndim == 4:
                data = torch.unsqueeze(data, 1)
            if self.args.ifalign_space:
                x_ori_list.append(self.linear_align(self.backbone_encoder[i](data)))
            else:
                x_ori_list.append(self.backbone_encoder[i](data))

        if self.modal_num == 3:
            x_0_ori = x_ori_list[0]
            x_1_ori = x_ori_list[1]
            x_2_ori = x_ori_list[2]

            # 得到 share 特征
            x_0_share = self.linear_dis_share(x_0_ori)
            x_1_share = self.linear_dis_share(x_1_ori)
            x_2_share = self.linear_dis_share(x_2_ori)
            x_share = (x_0_share + x_1_share + x_2_share) / self.modal_num

            # 得到 spe 特征
            x_0_spec = self.linear_dis_spec[0](x_0_ori)
            x_1_spec = self.linear_dis_spec[1](x_1_ori)
            x_2_spec = self.linear_dis_spec[2](x_2_ori)

            # 得到 sup 特征
            if self.args.ifdis_sup:
                x_01_sup_0 = self.linear_dis_sup[0](x_0_ori)
                x_01_sup_1 = self.linear_dis_sup[0](x_1_ori)
                x_01_sup = (x_01_sup_0 + x_01_sup_1) / 2

                x_02_sup_0 = self.linear_dis_sup[1](x_0_ori)
                x_02_sup_2 = self.linear_dis_sup[1](x_2_ori)
                x_02_sup = (x_02_sup_0 + x_02_sup_2) / 2

                x_12_sup_1 = self.linear_dis_sup[2](x_1_ori)
                x_12_sup_2 = self.linear_dis_sup[2](x_2_ori)
                x_12_sup = (x_12_sup_1 + x_12_sup_2) / 2

                if self.args.fusion_type == 'concat':
                    x_fusion = torch.cat((x_share, x_01_sup, x_02_sup, x_12_sup, x_0_spec, x_1_spec, x_2_spec), dim=1)
                elif self.args.fusion_type == 'moe':
                    x_fusion = self.moe_model([x_share, x_01_sup, x_02_sup, x_12_sup, x_0_spec, x_1_spec, x_2_spec])['moe_output']
                    # print(x_fusion.shape)
                    gating_weights = self.moe_model([x_share, x_01_sup, x_02_sup, x_12_sup, x_0_spec, x_1_spec, x_2_spec])['gating_weights']
            else:
                if self.args.fusion_type == 'concat':
                    x_fusion = torch.cat((x_share, x_0_spec, x_1_spec, x_2_spec), dim=1)
                elif self.args.fusion_type == 'moe':
                    x_fusion = self.moe_model([x_share, x_0_spec, x_1_spec, x_2_spec])['moe_output']
                    gating_weights = self.moe_model([x_share, x_0_spec, x_1_spec, x_2_spec])['gating_weights']

        elif self.modal_num == 4:
            x_0_ori = x_ori_list[0]
            x_1_ori = x_ori_list[1]
            x_2_ori = x_ori_list[2]
            x_3_ori = x_ori_list[3]

            # 得到 share 特征
            x_0_share = self.linear_dis_share(x_0_ori)
            x_1_share = self.linear_dis_share(x_1_ori)
            x_2_share = self.linear_dis_share(x_2_ori)
            x_3_share = self.linear_dis_share(x_3_ori)
            x_share = (x_0_share + x_1_share + x_2_share + x_3_share) / self.modal_num

            # 得到 spe 特征
            x_0_spec = self.linear_dis_spec[0](x_0_ori)
            x_1_spec = self.linear_dis_spec[1](x_1_ori)
            x_2_spec = self.linear_dis_spec[2](x_2_ori)
            x_3_spec = self.linear_dis_spec[3](x_3_ori)

            # 得到 sup 特征
            if self.args.ifdis_sup:
                # two modals
                if self.if_allpair:
                    x_01_sup_0 = self.linear_dis_sup[0](x_0_ori)
                    x_01_sup_1 = self.linear_dis_sup[0](x_1_ori)
                    x_01_sup = (x_01_sup_0 + x_01_sup_1) / 2

                    x_02_sup_0 = self.linear_dis_sup[1](x_0_ori)
                    x_02_sup_2 = self.linear_dis_sup[1](x_2_ori)
                    x_02_sup = (x_02_sup_0 + x_02_sup_2) / 2

                    x_03_sup_0 = self.linear_dis_sup[2](x_0_ori)
                    x_03_sup_3 = self.linear_dis_sup[2](x_3_ori)
                    x_03_sup = (x_03_sup_0 + x_03_sup_3) / 2

                    x_12_sup_1 = self.linear_dis_sup[3](x_1_ori)
                    x_12_sup_2 = self.linear_dis_sup[3](x_2_ori)
                    x_12_sup = (x_12_sup_1 + x_12_sup_2) / 2

                    x_13_sup_1 = self.linear_dis_sup[4](x_1_ori)
                    x_13_sup_3 = self.linear_dis_sup[4](x_3_ori)
                    x_13_sup = (x_13_sup_1 + x_13_sup_3) / 2

                    x_23_sup_2 = self.linear_dis_sup[5](x_2_ori)
                    x_23_sup_3 = self.linear_dis_sup[5](x_3_ori)
                    x_23_sup = (x_23_sup_2 + x_23_sup_3) / 2
                else:
                    x_01_sup_0 = self.linear_dis_sup[0](x_0_ori)
                    x_01_sup_1 = self.linear_dis_sup[0](x_1_ori)
                    x_01_sup = (x_01_sup_0 + x_01_sup_1) / 2

                    x_03_sup_0 = self.linear_dis_sup[1](x_0_ori)
                    x_03_sup_3 = self.linear_dis_sup[1](x_3_ori)
                    x_03_sup = (x_03_sup_0 + x_03_sup_3) / 2

                    x_12_sup_1 = self.linear_dis_sup[2](x_1_ori)
                    x_12_sup_2 = self.linear_dis_sup[2](x_2_ori)
                    x_12_sup = (x_12_sup_1 + x_12_sup_2) / 2

                    x_23_sup_2 = self.linear_dis_sup[3](x_2_ori)
                    x_23_sup_3 = self.linear_dis_sup[3](x_3_ori)
                    x_23_sup = (x_23_sup_2 + x_23_sup_3) / 2

                # three modals
                x_012_sup_0 = self.linear_dis_sup[4](x_0_ori)
                x_012_sup_1 = self.linear_dis_sup[4](x_1_ori)
                x_012_sup_2 = self.linear_dis_sup[4](x_2_ori)
                x_012_sup = (x_012_sup_0 + x_012_sup_1 + x_012_sup_2) / 3

                x_013_sup_0 = self.linear_dis_sup[5](x_0_ori)
                x_013_sup_1 = self.linear_dis_sup[5](x_1_ori)
                x_013_sup_3 = self.linear_dis_sup[5](x_3_ori)
                x_013_sup = (x_013_sup_0 + x_013_sup_1 + x_013_sup_3) / 3

                x_023_sup_0 = self.linear_dis_sup[6](x_0_ori)
                x_023_sup_2 = self.linear_dis_sup[6](x_2_ori)
                x_023_sup_3 = self.linear_dis_sup[6](x_3_ori)
                x_023_sup = (x_023_sup_0 + x_023_sup_2 + x_023_sup_3) / 3

                x_123_sup_1 = self.linear_dis_sup[7](x_1_ori)
                x_123_sup_2 = self.linear_dis_sup[7](x_2_ori)
                x_123_sup_3 = self.linear_dis_sup[7](x_3_ori)
                x_123_sup = (x_123_sup_1 + x_123_sup_2 + x_123_sup_3) / 3

                if self.args.fusion_type == 'concat':
                    if self.if_allpair:
                        x_fusion = torch.cat((x_share,
                                              x_01_sup, x_02_sup, x_03_sup, x_12_sup, x_13_sup, x_23_sup,
                                              x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                              x_0_spec, x_1_spec, x_2_spec, x_3_spec), dim=1)
                    else:
                        x_fusion = torch.cat((x_share,
                                              x_01_sup, x_03_sup, x_12_sup, x_23_sup,
                                              x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                              x_0_spec, x_1_spec, x_2_spec, x_3_spec), dim=1)
                elif self.args.fusion_type == 'moe':
                    if self.if_allpair:
                        x_fusion = self.moe_model([x_share,
                                                   x_01_sup, x_02_sup, x_03_sup, x_12_sup, x_13_sup, x_23_sup,
                                                   x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                                   x_0_spec, x_1_spec, x_2_spec, x_3_spec])['moe_output']
                        gating_weights = self.moe_model([x_share,
                                                         x_01_sup, x_02_sup, x_03_sup, x_12_sup, x_13_sup, x_23_sup,
                                                         x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                                         x_0_spec, x_1_spec, x_2_spec, x_3_spec])['gating_weights']
                    else:
                        x_fusion = self.moe_model([x_share,
                                                   x_01_sup, x_03_sup, x_12_sup, x_23_sup,
                                                   x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                                   x_0_spec, x_1_spec, x_2_spec, x_3_spec])['moe_output']

                        gating_weights = self.moe_model([x_share,
                                                         x_01_sup, x_03_sup, x_12_sup, x_23_sup,
                                                         x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                                         x_0_spec, x_1_spec, x_2_spec, x_3_spec])['gating_weights']
            else:
                if self.args.fusion_type == 'concat':
                    x_fusion = torch.cat((x_share, x_0_spec, x_1_spec, x_2_spec, x_3_spec), dim=1)
                elif self.args.fusion_type == 'moe':
                    x_fusion = self.moe_model([x_share, x_0_spec, x_1_spec, x_2_spec, x_3_spec])['moe_output']
                    gating_weights = self.moe_model([x_share, x_0_spec, x_1_spec, x_2_spec, x_3_spec])['gating_weights']

        out_prediction = self.linear_cls(x_fusion)

        if self.modal_num == 4 and (not self.if_allpair):
            x_02_sup = False
            x_13_sup = False

        if run_type == 'test':
            if self.args.ifdis_sup:
                if self.args.fusion_type == 'concat':
                    gating_weights = None
                if self.modal_num == 3:
                    return {'out_prediction': out_prediction,
                            'x_0_ori': x_0_ori,
                            'x_1_ori': x_1_ori,
                            'x_2_ori': x_2_ori,
                            'x_0_share': x_0_share,
                            'x_1_share': x_1_share,
                            'x_2_share': x_2_share,
                            'x_share': x_share,
                            'x_01_sup': x_01_sup,
                            'x_02_sup': x_02_sup,
                            'x_12_sup': x_12_sup,
                            'x_01_sup_0': x_01_sup_0,
                            'x_01_sup_1': x_01_sup_1,
                            'x_02_sup_0': x_02_sup_0,
                            'x_02_sup_2': x_02_sup_2,
                            'x_12_sup_1': x_12_sup_1,
                            'x_12_sup_2': x_12_sup_2,
                            'x_0_spec': x_0_spec,
                            'x_1_spec': x_1_spec,
                            'x_2_spec': x_2_spec,
                            'gating_weights': gating_weights}
                elif self.modal_num == 4:
                    return {'out_prediction': out_prediction,
                            'x_0_ori': x_0_ori,
                            'x_1_ori': x_1_ori,
                            'x_2_ori': x_2_ori,
                            'x_3_ori': x_3_ori,
                            'x_0_share': x_0_share,
                            'x_1_share': x_1_share,
                            'x_2_share': x_2_share,
                            'x_3_share': x_3_share,
                            'x_share': x_share,
                            'x_01_sup': x_01_sup,
                            'x_02_sup': x_02_sup,
                            'x_03_sup': x_03_sup,
                            'x_12_sup': x_12_sup,
                            'x_13_sup': x_13_sup,
                            'x_23_sup': x_23_sup,
                            'x_012_sup': x_012_sup,
                            'x_013_sup': x_013_sup,
                            'x_023_sup': x_023_sup,
                            'x_123_sup': x_123_sup,
                            'x_01_sup_0': x_01_sup_0,
                            'x_01_sup_1': x_01_sup_1,
                            'x_03_sup_0': x_03_sup_0,
                            'x_03_sup_3': x_03_sup_3,
                            'x_12_sup_1': x_12_sup_1,
                            'x_12_sup_2': x_12_sup_2,
                            'x_23_sup_2': x_23_sup_2,
                            'x_23_sup_3': x_23_sup_3,
                            'x_012_sup_0': x_012_sup_0,
                            'x_012_sup_1': x_012_sup_1,
                            'x_012_sup_2': x_012_sup_2,
                            'x_013_sup_0': x_013_sup_0,
                            'x_013_sup_1': x_013_sup_1,
                            'x_013_sup_3': x_013_sup_3,
                            'x_023_sup_0': x_023_sup_0,
                            'x_023_sup_2': x_023_sup_2,
                            'x_023_sup_3': x_023_sup_3,
                            'x_123_sup_1': x_123_sup_1,
                            'x_123_sup_2': x_123_sup_2,
                            'x_123_sup_3': x_123_sup_3,
                            'x_0_spec': x_0_spec,
                            'x_1_spec': x_1_spec,
                            'x_2_spec': x_2_spec,
                            'x_3_spec': x_3_spec,
                            'gating_weights': gating_weights}
            else:
                if self.args.fusion_type == 'concat':
                    gating_weights = None

                if self.modal_num == 3:
                    return {'out_prediction': out_prediction,
                            'x_0_ori': x_0_ori,
                            'x_1_ori': x_1_ori,
                            'x_2_ori': x_2_ori,
                            'x_0_share': x_0_share,
                            'x_1_share': x_1_share,
                            'x_2_share': x_2_share,
                            'x_0_spec': x_0_spec,
                            'x_1_spec': x_1_spec,
                            'x_2_spec': x_2_spec,
                            'gating_weights': gating_weights}
                elif self.modal_num == 4:
                    return {'out_prediction': out_prediction,
                            'x_0_ori': x_0_ori,
                            'x_1_ori': x_1_ori,
                            'x_2_ori': x_2_ori,
                            'x_3_ori': x_3_ori,
                            'x_0_share': x_0_share,
                            'x_1_share': x_1_share,
                            'x_2_share': x_2_share,
                            'x_3_share': x_3_share,
                            'x_0_spec': x_0_spec,
                            'x_1_spec': x_1_spec,
                            'x_2_spec': x_2_spec,
                            'x_3_spec': x_3_spec,
                            'gating_weights': gating_weights}

        if self.args.run_type == 'get_param':
            return {'out_prediction': out_prediction}

        ### loss ###
        # main cls loss
        main_cls_loss = torch.mean(self.criterion(out_prediction, label))*self.args.w_main_cls

        if self.modal_num == 3:
            # share sim loss
            share_sim_loss = self.sim_operation_between_3(x_0_share,x_1_share,x_2_share)*self.args.w_sim

            if self.args.ifdis_sup:
                # sup sim loss
                sup_sim_loss = (self.calc_sim(x_01_sup_0, x_01_sup_1) + \
                                self.calc_sim(x_02_sup_0, x_02_sup_2) + \
                                self.calc_sim(x_12_sup_1, x_12_sup_2))*self.args.w_sim
                diff_list = [x_share, x_01_sup, x_02_sup, x_12_sup, x_0_spec, x_1_spec, x_2_spec]
            else:
                sup_sim_loss = torch.tensor([0.]).cuda()
                diff_list = [x_share, x_0_spec, x_1_spec, x_2_spec]

            # # recon loss
            # if self.args.ifrecon:
            #     if self.args.ifdis_sup:
            #         recon_loss = self.recon_op(x_ori_list, x_share_list, x_share_sup_list, x_dif_list) * self.args.w_recon
            #     else:
            #         recon_loss = self.recon_op(x_ori_list=x_ori_list, x_share_list=x_share_list, x_dif_list=x_dif_list)*self.args.w_recon
            # else:
            #     recon_loss = torch.tensor([0]).cuda()
        elif self.modal_num == 4:
            # share sim loss
            share_sim_loss = (self.calc_sim(x_0_share,x_1_share) + \
                              self.calc_sim(x_0_share,x_2_share) + \
                              self.calc_sim(x_0_share,x_3_share) + \
                              self.calc_sim(x_1_share,x_2_share) + \
                              self.calc_sim(x_1_share,x_3_share) + \
                              self.calc_sim(x_2_share,x_3_share))*self.args.w_sim

            if self.args.ifdis_sup:
                # sup sim loss
                if self.if_allpair:
                    sup_sim_loss = (self.calc_sim(x_01_sup_0, x_01_sup_1) + \
                                    self.calc_sim(x_02_sup_0, x_02_sup_2) + \
                                    self.calc_sim(x_03_sup_0, x_03_sup_3) + \
                                    self.calc_sim(x_12_sup_1, x_12_sup_2) + \
                                    self.calc_sim(x_13_sup_1, x_13_sup_3) + \
                                    self.calc_sim(x_23_sup_2, x_23_sup_3) + \
                                    self.sim_operation_between_3(x_012_sup_0, x_012_sup_1, x_012_sup_2) + \
                                    self.sim_operation_between_3(x_013_sup_0, x_013_sup_1, x_013_sup_3) + \
                                    self.sim_operation_between_3(x_023_sup_0, x_023_sup_2, x_023_sup_3) + \
                                    self.sim_operation_between_3(x_123_sup_1, x_123_sup_2,
                                                                 x_123_sup_3)) * self.args.w_sim
                    diff_list = [x_share,
                                 x_01_sup, x_02_sup, x_03_sup, x_12_sup, x_13_sup, x_23_sup,
                                 x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                 x_0_spec, x_1_spec, x_2_spec, x_3_spec]
                else:
                    sup_sim_loss = (self.calc_sim(x_01_sup_0, x_01_sup_1) + \
                                    self.calc_sim(x_03_sup_0, x_03_sup_3) + \
                                    self.calc_sim(x_12_sup_1, x_12_sup_2) + \
                                    self.calc_sim(x_23_sup_2, x_23_sup_3) + \
                                    self.sim_operation_between_3(x_012_sup_0, x_012_sup_1, x_012_sup_2) + \
                                    self.sim_operation_between_3(x_013_sup_0, x_013_sup_1, x_013_sup_3) + \
                                    self.sim_operation_between_3(x_023_sup_0, x_023_sup_2, x_023_sup_3) + \
                                    self.sim_operation_between_3(x_123_sup_1, x_123_sup_2, x_123_sup_3))*self.args.w_sim
                    diff_list = [x_share,
                                 x_01_sup, x_03_sup, x_12_sup, x_23_sup,
                                 x_012_sup, x_013_sup, x_023_sup, x_123_sup,
                                 x_0_spec, x_1_spec, x_2_spec, x_3_spec]
            else:
                sup_sim_loss = torch.tensor([0.]).cuda()
                diff_list = [x_share, x_0_spec, x_1_spec, x_2_spec, x_3_spec]

            # # recon loss
            # if self.args.ifrecon:
            #     if self.args.ifdis_sup:
            #         recon_loss = self.recon_op(x_ori_list, x_share_list, x_share_sup_list, x_dif_list) * self.args.w_recon
            #     else:
            #         recon_loss = self.recon_op(x_ori_list=x_ori_list, x_share_list=x_share_list, x_dif_list=x_dif_list)*self.args.w_recon
            # else:
            #     recon_loss = torch.tensor([0]).cuda()

        # diff loss
        diff_loss = self.calc_diff_loss(diff_list)*self.args.w_diff

        moe_aux_loss = torch.tensor([0.]).cuda()
        if self.args.ifmoe_aux:
            for i in range(len(diff_list)):
                moe_aux_loss += torch.mean(self.criterion(self.linear_cls_aux[i](diff_list[i]), label))*self.args.w_moe_aux

        # contrastive center loss
        if self.args.fusion_type == 'moe' and self.args.if_ccl:
            cc_loss = self.CC_Loss(gating_weights,label)*self.args.w_ccl
        else:
            cc_loss = torch.tensor([0.]).cuda()

        loss = main_cls_loss + share_sim_loss + sup_sim_loss + diff_loss + moe_aux_loss + cc_loss

        return {'out_prediction': out_prediction,
                'loss': loss,
                'main_cls_loss': main_cls_loss,
                'share_sim_loss': share_sim_loss,
                'sup_sim_loss': sup_sim_loss,
                'diff_loss': diff_loss,
                'cc_loss': cc_loss,
                'moe_aux_loss': moe_aux_loss}

