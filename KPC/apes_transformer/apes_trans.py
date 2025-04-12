import torch
from torch import nn
from apes_transformer import ops
from einops import rearrange, repeat
import math
from models.KPC_util import farthest_point_sample,index_points,sample_and_group
import torch.nn.functional as F
from apes_transformer.mulit_fu import PointTransformerLayer

class slefAttention(nn.Module):
    def __init__(self,in_channels):
        super(slefAttention, self).__init__()
        self.q_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.k_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.v_conv = nn.Conv1d(in_channels, in_channels, 1, bias=False)
        self.pos = nn.Conv1d(3, in_channels, 1, bias=False)
        self.pos1 = nn.Conv1d(3, in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(normalized_shape=512)
    def forward(self,x1,p1,x2,p2):
        _,_,n = x1.size()
        pos = self.pos(p1)
        pos1 = self.pos(p2)
        q = self.q_conv(x1+pos)  # (B, C, N) -> (B, C, N)
        k = self.k_conv(x2+pos1)  # (B, C, N) -> (B, C, N)
        v = self.v_conv(x2)  # (B, C, N) -> (B, C, N)
        energy = rearrange(q, 'B C N -> B N C').contiguous() @ k  # (B, N, C) @ (B, C, N) -> (B, N, N)
        scale_factor = math.sqrt(q.shape[-2])
        attention = self.softmax(energy / scale_factor)  # (B, N, N) -> (B, N, N)
        v = attention @ rearrange(v, 'B C N -> B N C').contiguous()  # (B, M, N) @ (B, N, C) -> (B, M, C)
        out = rearrange(v, 'B M C -> B C M').contiguous()  # (B, M, C) -> (B, C, M)
        out = out + x1
        return out.transpose(1,2)

class slefAttentionMutil(nn.Module):
    def __init__(self,in_channels):
        super(slefAttentionMutil, self).__init__()
        self.heads = 4
        self.q_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.k_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.v_conv = nn.Conv2d(in_channels, in_channels, 1, bias=False)
        self.pos = nn.Conv1d(3, in_channels, 1, bias=False)
        self.pos1 = nn.Conv1d(3, in_channels, 1, bias=False)
        self.softmax = nn.Softmax(dim=-1)
        self.norm = nn.LayerNorm(normalized_shape=512)

    def forward(self,x,x1,p,p1):
        _, _, n = x.size()
        pos = self.pos(p).unsqueeze(-1)
        pos1 = self.pos1(p1).unsqueeze(-1)
        q = self.q_conv(rearrange(x, 'B C N -> B C N 1')).contiguous()  # (B, C, N) -> (B, C, N, 1)
        q = torch.add(q,pos)
        q = self.split_heads(q, self.heads)  # (B, C, N, 1) -> (B, H, N, 1, D)

        k = self.k_conv(rearrange(x1, 'B C N -> B C N 1')).contiguous()  # (B, C, N, K) -> (B, C, N, K)
        k = torch.add(k,pos1)
        k = self.split_heads(k, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)

        v = self.v_conv(rearrange(x1, 'B C N -> B C N 1')).contiguous()  # (B, C, N, K) -> (B, C, N, K)
        v = torch.add(v, pos1)
        v = self.split_heads(v, self.heads)  # (B, C, N, K) -> (B, H, N, K, D)

        energy = q @ rearrange(k,
                               'B H N K D -> B H N D K').contiguous()  # (B, H, N, 1, D) @ (B, H, N, D, K) -> (B, H, N, 1, K)
        scale_factor = math.sqrt(q.shape[-1])
        attention = self.softmax(energy / scale_factor)  # (B, H, N, 1, K) -> (B, H, N, 1, K)
        tmp = rearrange(attention @ v, 'B H N 1 D -> B (H D) N').contiguous()
        return tmp

    @staticmethod
    def split_heads(x, heads):
        x = rearrange(x, 'B (H D) N K -> B H N K D', H=heads).contiguous()  # (B, C, N, K) -> (B, H, N, K, D)
        return x

class pointDiff(nn.Module):
    def __init__(self,inchannels,outchannels,in_channels):
        super(pointDiff, self).__init__()
        self.mlp = nn.Sequential(
        nn.BatchNorm1d(inchannels),
        nn.ReLU(),
        nn.Conv1d(inchannels, inchannels, 1, bias=False),
        nn.Sigmoid()
        )
        self.mlp1 = nn.Sequential(
            nn.BatchNorm1d(inchannels),
            nn.ReLU(),
            nn.Conv1d(inchannels, inchannels, 1, bias=False),
            nn.Sigmoid()
        )
        self.mlp_gate = nn.Sequential(
            nn.BatchNorm1d(inchannels),
            nn.Conv1d(inchannels, inchannels, 1, bias=False),
            nn.Sigmoid()
        )
        self.diff = nn.Sequential(
            nn.Conv1d(in_channels, outchannels, 1, bias=False),
            nn.BatchNorm1d(outchannels),
            nn.ReLU(),
            nn.Conv1d(outchannels, outchannels, 1, bias=False)
        )
    def forward(self, conv_ly, trans_ly, origin_f):
        b,c,n = trans_ly.size()
        conv = self.mlp(conv_ly)
        trans = self.mlp1(trans_ly)
        x = self.diff(origin_f)
        origin = F.interpolate(x, size=n, mode='linear', align_corners=False)
        z = self.mlp_gate(conv + trans)

        tmp = (1 - z) * conv + z * trans
        tmp = tmp.view(b, c, n)
        tmp = tmp + origin
        return tmp

class im_choose(nn.Module):
    def __init__(self,inchannel_1,outchannel):
        super(im_choose, self).__init__()

        self.q1 = nn.Conv1d(inchannel_1,outchannel,1,bias=False)
        self.k1 = nn.Conv1d(inchannel_1,outchannel,1,bias=False)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,l1,xyz1,top_k):
        q1 = self.q1(l1)
        k1 = self.k1(l1)
        energy1 = rearrange(q1, 'B C N -> B N C').contiguous() @ k1  # (B, N, C) @ (B, C, M) -> (B, N, M)
        scale_factor1 = math.sqrt(q1.shape[-2])
        attention1 = self.softmax(energy1 / scale_factor1)  # (B, N, M) -> (B, N, M)
        att_se1 = attention1[:,0,:]
        _, topk_indices1 = torch.topk(att_se1, k=top_k,dim=-1)
        expanded_indices1 = topk_indices1.unsqueeze(1).expand(-1, l1.shape[1], -1)
        expanded_indices_pos1 = topk_indices1.unsqueeze(1).expand(-1, 3, -1)
        l1_new = torch.gather(l1, dim=-1, index=expanded_indices1)
        p1_new = torch.gather(xyz1, dim=-1, index=expanded_indices_pos1)

        return l1_new.transpose(1,2),p1_new.transpose(1,2)

class IMFF(nn.Module):
    def __init__(self,in_channel,out_channel,real_out):
        super(IMFF, self).__init__()
        self.choose = im_choose(in_channel,out_channel)
        # self.pointrans = PointTransformerLayer(out_channel,out_channel,share_channel,nsample,l2_point,radius)
        self.pointtrans = slefAttention(out_channel)
        self.norm = nn.LayerNorm(out_channel)
        self.mlp = nn.Sequential(nn.Linear(in_channel,in_channel),
                                 nn.LayerNorm(in_channel),
                                 nn.ReLU(),
                                 nn.Linear(in_channel,out_channel))

        self.mlp1 = nn.Sequential(nn.Linear(out_channel,out_channel),
                                 nn.LayerNorm(out_channel),
                                 nn.ReLU(),
                                 nn.Linear(out_channel,real_out))
    def forward(self,l1,xyz1,l2,xyz2):
        top_k = l2.shape[2]
        l1_new,xyz1_new = self.choose(l1,xyz1,top_k)
        l1_new = self.mlp(l1_new)
        l1_new,xyz1_new = l1_new.transpose(1,2),xyz1_new.transpose(1,2)
        l2_new = self.pointtrans(l2,xyz2,l1_new,xyz1_new)
        l2_new = F.leaky_relu(self.norm(l2_new))
        l2_new = self.mlp1(l2_new)
        return l2_new.transpose(1,2)

# input_tensor = torch.randn(16, 128, 128)
# p1 = torch.randn(16, 3, 128)
# input_tensor_1 = torch.randn(16, 256, 64)
# p2 = torch.randn(16, 3, 64)
# model = IMFF(128,256,128)
# out = model(input_tensor,p1,input_tensor_1,p2)
# print(1)

