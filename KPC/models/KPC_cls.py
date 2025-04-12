'''
@Project ：pointnet_walk_1 
@File    ：KPC_cls.py
@IDE     ：PyCharm 
@Author  ：liangdegao
@Date    ：2025/3/17 13:40 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction,PointNetFeaturePropagation
from apes_transformer.apes_trans import IMFF
import math

class curve_feature(nn.Module):
    def __init__(self, in_channel):
        super(curve_feature, self).__init__()
        self.x0_mlp = nn.Sequential(
            nn.Conv2d(in_channel,
                      32,
                      kernel_size=1,
                      bias=False),
            nn.Conv2d(32,
                      64,
                      kernel_size=1,
                      bias=False),
        )
        self.x1_mlp = nn.Sequential(
            nn.Conv2d(in_channel,
                      64,
                      kernel_size=1,
                      bias=False),
            nn.Conv2d(64,
                      128,
                      kernel_size=1,
                      bias=False),
        )
        self.x2_mlp = nn.Sequential(
            nn.Conv2d(in_channel,
                      64,
                      kernel_size=1,
                      bias=False),
            nn.Conv2d(64,
                      256,
                      kernel_size=1,
                      bias=False),
        )
        self.x3_mlp = nn.Sequential(
            nn.Conv2d(in_channel,
                      128,
                      kernel_size=1,
                      bias=False),
            nn.Conv2d(128,
                      512,
                      kernel_size=1,
                      bias=False),
        )
    def forward(self, group):
        b,_,_,_ = group.size()

        b = int(b / 2)
        group_1 = group[:b]
        group_2 = group[-b:]

        x_64 = F.relu(self.x0_mlp(group_1))

        x_128 = F.relu(self.x1_mlp(group_1))

        x_256 = F.relu(self.x2_mlp(group_2))

        x_512 = F.relu(self.x3_mlp(group_2))
        return x_64,x_128,x_256,x_512

class KPC_fu(nn.Module):
    def __init__(self, in_channel):
        super(KPC_fu, self).__init__()
        self.in_channel = in_channel
        mid_feature = in_channel // 2
        self.conva = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)

        self.conva_v = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)

        self.convb = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convb_v = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convc = nn.Conv1d(in_channel,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convn = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convl = nn.Conv1d(mid_feature,
                               mid_feature,
                               kernel_size=1,
                               bias=False)
        self.convd = nn.Sequential(
            nn.Conv1d(mid_feature * 2,
                      in_channel,
                      kernel_size=1,
                      bias=False),
            nn.BatchNorm1d(in_channel))
        self.line_conv_att = nn.Conv2d(in_channel,
                                       1,
                                       kernel_size=1,
                                       bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(in_channel)

        self.p_l = nn.Conv1d(3, mid_feature, 1)
        self.p_n = nn.Conv1d(3, mid_feature, 1)
    def forward(self, x, curves):
        p, curve = curves

        p_l = torch.mean(self.p_l(p), dim=-1)
        p_n = torch.mean(self.p_n(p), dim=-2)

        curves_att = self.line_conv_att(curves)  # bs, 1, c_n, c_l
        curver_inter = torch.sum(curves * F.softmax(curves_att, dim=-1), dim=-1)  # bs, c, c_n
        curves_intra = torch.sum(curves * F.softmax(curves_att, dim=-2), dim=-2)  # bs, c, c_l

        k_inter = self.conva(curver_inter + p_l)  # bs, mid, n
        v_inter = self.conva_v(curver_inter + p_l).transpose(1,2)

        k_intra = self.convb(curves_intra + p_n)  # bs, mid ,n
        v_intra = self.convb_v(curves_intra + p_n).transpose(1,2)

        x_logits_q = self.convc(x).transpose(1, 2).contiguous()

        en_inter = torch.bmm(x_logits_q, k_inter)
        scale_factor1 = math.sqrt(x_logits_q.shape[-1])
        att_inter = self.softmax(en_inter / scale_factor1)  # (B, N, M) -> (B, N, M)
        feature_inter = torch.bmm(att_inter,v_inter).contiguous()

        en_intra = torch.bmm(x_logits_q, k_intra)
        scale_factor2 = math.sqrt(x_logits_q.shape[-1])
        att_intra = self.softmax(en_intra / scale_factor2)
        feature_intra = torch.bmm(att_intra, v_intra).contiguous()

        curve_features = self.norm(torch.cat((feature_inter, feature_intra),dim=-1)).transpose(1,2).contiguous()
        x = x + self.convd(curve_features)

        return F.leaky_relu(x, negative_slope=0.2)

class get_model_cls(nn.Module):
    def __init__(self, num_classes, in_channel, channel=16):
        super(get_model_cls, self).__init__()
        planes = [512,256,128,64]
        self.sa1 = PointNetSetAbstraction(512, 0.1, 32, 3 + 3, [32, 32, 64], False)
        self.sa2 = PointNetSetAbstraction(128, 0.2, 32, 64 + 3, [64, 64, 128], False)
        self.sa3 = PointNetSetAbstraction(64, 0.4, 32, 128 + 3, [128, 128, 128], False)
        self.sa4 = PointNetSetAbstraction(16, 0.8, 32, 128 + 3, [256, 256, 256], False)

        self.conv0 = nn.Sequential(
            nn.Conv1d(256, 1024, kernel_size=1, bias=False),
            nn.BatchNorm1d(1024),
            nn.ReLU(inplace=True))
        self.conv1 = nn.Linear(1024 * 2, 512, bias=False)
        self.conv2 = nn.Linear(512, num_classes)
        self.bn1 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=0.3)

        self.imff3 = IMFF(256, 512, 256)

        self.l1 = nn.Sequential(nn.Linear(in_channel[3], in_channel[3] // 2),
                                nn.LayerNorm(in_channel[3] // 2),
                                nn.ReLU(),
                                nn.Linear(in_channel[3] // 2, in_channel[3] // 2))
    def forward(self, xyz, group):
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        if self.training:
            x_64, x_128, x_256, x_512 = self.curve_freature(group)
            l1_points = self.curve_agg_0(l1_points, x_64)
            l2_points = self.curve_agg_1(l2_points, x_128)
            l3_points = self.curve_agg_2(l3_points, x_256)
            l4_points = self.curve_agg_3(l4_points, x_512)

        l4_points_1 = self.imff3(l3_points, l3_xyz, l4_points, l4_xyz)
        l4_points = torch.cat((self.l1(l4_points.transpose(1, 2)).transpose(1, 2), l4_points_1), dim=1)

        x = self.conv0(l4_points)
        x_max = F.adaptive_max_pool1d(x, 1)
        x_avg = F.adaptive_avg_pool1d(x, 1)

        x = torch.cat((x_max, x_avg), dim=1).squeeze(-1)
        x = F.relu(self.bn1(self.conv1(x).unsqueeze(-1)), inplace=True).squeeze(-1)
        x = self.dp1(x)
        x = self.conv2(x)

        return x