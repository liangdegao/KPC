import torch
from torch import nn
from apes_transformer import ops
import math
from models.KPC_util import farthest_point_sample,index_points,sample_and_group_1,index_points
import torch.nn.functional as F
import einops


class PointTransformerLayer(nn.Module):
    def __init__(self, in_planes, out_planes, share_planes, nsample, down_point, radius):
        super().__init__()
        self.mid_planes = mid_planes = out_planes // 2
        self.out_planes = out_planes
        self.share_planes = share_planes
        self.nsample = nsample
        self.down_point = down_point
        self.radius = radius
        self.linear_q = nn.Linear(in_planes, mid_planes)
        self.linear_k = nn.Linear(in_planes, mid_planes)
        self.linear_v = nn.Linear(in_planes, out_planes)
        self.linear_p = nn.Sequential(nn.Linear(3, 3),
                                      nn.LayerNorm(3),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes))
        self.linear_p_1 = nn.Sequential(nn.Linear(3, 3),
                                      nn.LayerNorm(3),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(3, out_planes//2))
        self.linear_w = nn.Sequential(nn.LayerNorm(mid_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(mid_planes, out_planes // share_planes),
                                      nn.LayerNorm(out_planes // share_planes),
                                      nn.ReLU(inplace=True),
                                      nn.Linear(out_planes // share_planes, out_planes // share_planes))
        self.softmax = nn.Softmax(dim=1)


    def forward(self, x, xyz, x1,xyz1):
        x_q, x_k, x_v = self.linear_q(x.transpose(1,2)), self.linear_k(x1), self.linear_v(x1)
        new_xyz, x_k, idx = sample_and_group_1(self.down_point,self.radius,self.nsample,xyz1, x_k)
        x_v = index_points(x_v, idx)
        p_r, x_k = x_k[:, :, :, 0:3], x_k[:, :, :, 3:]
        p_r = self.linear_p(p_r)
        p_r_1 = self.linear_p_1(xyz.transpose(1,2))
        r_qk = x_k - (x_q + p_r_1).unsqueeze(2) + einops.reduce(p_r, "b n ns (i j) -> b n ns j", reduction="sum", j=self.mid_planes)
        w = self.linear_w(r_qk)  # (n, nsample, c)
        w = self.softmax(w)
        x = torch.einsum("b n t s i, b n t i -> b n s i",
                         einops.rearrange(x_v + p_r, "b n ns (s i) ->b n ns s i", s=self.share_planes), w)
        x = einops.rearrange(x, "b n s i -> b n (s i)")
        return x,new_xyz
