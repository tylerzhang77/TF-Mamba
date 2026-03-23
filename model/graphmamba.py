# model/graphmamba.py

import math
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from einops import rearrange, repeat
import torch.nn.functional as F
from torch_topological.nn.data import make_tensor
from torch_topological.nn import VietorisRipsComplex
from torch_topological.nn.layers import StructureElementLayer

from model.lib import ST_RenovateNet
from model.stmamba import ST_Mamba


def import_class(name):
    components = name.split('.')
    mod = __import__(components[0])
    for comp in components[1:]:
        mod = getattr(mod, comp)
    return mod


def conv_branch_init(conv, branches):
    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2.0 / (n * k1 * k2 * branches)))
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def conv_init(conv):
    if conv.weight is not None:
        nn.init.kaiming_normal_(conv.weight, mode='fan_out')
    if conv.bias is not None:
        nn.init.constant_(conv.bias, 0)


def bn_init(bn, scale):
    nn.init.constant_(bn.weight, scale)
    nn.init.constant_(bn.bias, 0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        if hasattr(m, 'weight'):
            nn.init.kaiming_normal_(m.weight, mode='fan_out')
        if hasattr(m, 'bias') and m.bias is not None and isinstance(m.bias, torch.Tensor):
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        if hasattr(m, 'weight') and m.weight is not None:
            m.weight.data.normal_(1.0, 0.02)
        if hasattr(m, 'bias') and m.bias is not None:
            m.bias.data.fill_(0)


class TemporalConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1):
        super(TemporalConv, self).__init__()
        pad = (kernel_size + (kernel_size - 1) * (dilation - 1) - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1),
            dilation=(dilation, 1))
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        return x


class unit_tcn(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1):
        super(unit_tcn, self).__init__()
        pad = int((kernel_size - 1) / 2)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, 1),
            padding=(pad, 0),
            stride=(stride, 1)
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        conv_init(self.conv)
        bn_init(self.bn, 1)

    def forward(self, x):
        x = self.bn(self.conv(x))
        return x


class TemporalMambaWrapper(nn.Module):
    """Temporal modeling with ST_Mamba."""
    def __init__(self,
                 in_channels,
                 out_channels,
                 mode='temporal',
                 d_state=4,
                 d_conv=3,
                 expand=1,
                 if_divide_out=False):
        super().__init__()
        assert in_channels == out_channels, "TemporalMambaWrapper requires Cin == Cout"
        self.mamba = ST_Mamba(
            dim_in=in_channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
            mode=mode,
            if_divide_out=if_divide_out,
        )
        self.ln = nn.LayerNorm(in_channels)

    def forward(self, x):
        x = x.permute(0, 2, 3, 1).contiguous()
        x = self.ln(x)
        with torch.cuda.amp.autocast(enabled=False):
            x = x.float()
            x = self.mamba(x)
        x = x.permute(0, 3, 1, 2).contiguous()
        return x


class unit_gcn(nn.Module):
    def __init__(self, in_channels, out_channels, A, adaptive=True, alpha=False):
        super(unit_gcn, self).__init__()
        self.out_c = out_channels
        self.in_c = in_channels
        self.num_heads = 8 if in_channels > 8 else 1
        self.fc1 = nn.Parameter(
            torch.stack(
                [torch.stack([torch.eye(A.shape[-1]) for _ in range(self.num_heads)], dim=0) for _ in range(3)],
                dim=0),
            requires_grad=True
        )
        self.fc2 = nn.ModuleList(
            [nn.Conv2d(in_channels, out_channels, 1, groups=self.num_heads) for _ in range(3)]
        )

        if in_channels != out_channels:
            self.down = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.down = lambda x: x

        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                conv_init(m)
            elif isinstance(m, nn.BatchNorm2d):
                bn_init(m, 1)
        bn_init(self.bn, 1e-6)

        h1 = A.sum(0)
        h1[h1 != 0] = 1

        h = [None for _ in range(A.shape[-1])]
        h[0] = np.eye(A.shape[-1])
        h[1] = h1
        self.hops = 0 * h[0]
        for i in range(2, A.shape[-1]):
            h[i] = h[i - 1] @ h1.transpose(0, 1)
            h[i][h[i] != 0] = 1

        for i in range(A.shape[-1] - 1, 0, -1):
            if np.any(h[i] - h[i - 1]):
                h[i] = h[i] - h[i - 1]
                self.hops += i * h[i]
            else:
                continue

        self.hops = torch.tensor(self.hops).long()
        self.rpe = nn.Parameter(torch.zeros((3, self.num_heads, self.hops.max() + 1,)))

        if alpha:
            self.alpha = nn.Parameter(torch.ones(1, self.num_heads, 1, 1, 1))
        else:
            self.alpha = 1

    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, 2, dim=-2, keepdim=True) + 1e-4
        return weight_norm

    def forward(self, x):
        N, C, T, V = x.size()
        y = None
        pos_emb = self.rpe[:, :, self.hops]
        for i in range(3):
            weight_norm = self.L2_norm(self.fc1[i])
            w1 = self.fc1[i] / weight_norm
            w1 = w1 + pos_emb[i] / self.L2_norm(pos_emb[i])
            x_in = x.view(N, self.num_heads, C // self.num_heads, T, V)
            z = torch.einsum("nhctv, hvw->nhctw", (x_in, w1)).contiguous().view(N, -1, T, V)
            z = self.fc2[i](z)
            y = z + y if y is not None else z
        y = self.bn(y)
        y += self.down(x)
        y = self.relu(y)
        return y


class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class TCN_GCN_unit_mamba(nn.Module):
    """GCN + SEBlock + (optional HHT dynamics injection) + ST_Mamba (temporal)"""
    def __init__(self, in_channels, out_channels, A, stride=1, residual=True, adaptive=True,
                 kernel_size=5, dilations=[1, 2], num_point=25, num_heads=16, alpha=False):
        super().__init__()
        self.gcn1 = unit_gcn(in_channels, out_channels, A, adaptive=adaptive, alpha=alpha)
        self.tcn1 = TemporalMambaWrapper(
            in_channels=out_channels,
            out_channels=out_channels,
            mode='temporal',
            d_state=16,
            d_conv=3,
            expand=2,
            if_divide_out=False,
        )
        self.relu = nn.ReLU(inplace=True)

        if not residual:
            self.residual = lambda x: 0
        elif in_channels == out_channels:
            self.residual = lambda x: x
        else:
            self.residual = unit_tcn(in_channels, out_channels, kernel_size=1, stride=1)

        self.se = SEBlock(out_channels, reduction=16)

    def forward(self, x, hht_ctv=None):
        """
        Args:
            x: [N, C_in, T, V]
            hht_ctv: optional [N, C_out, T, V] HHT prior (projected, broadcast over joints); added to gx after GCN+SE, then Mamba
        """
        x_in = x
        gx = self.gcn1(x_in)
        gx = self.se(gx)
        if hht_ctv is not None:
            gx = gx + hht_ctv
        tx = self.tcn1(gx)
        rx = self.residual(x_in)
        y = tx + rx
        return y


# --- TopoTrans (single-person, fixed layout) ---
class TopoTrans(nn.Module):
    """Topological feature transform (single-person, fixed variant)."""
    def __init__(self, out_dim):
        super(TopoTrans, self).__init__()
        self.relu = nn.ReLU()
        self.mlp = nn.Linear(64, out_dim)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, x):
        # x: [N, 64]
        x = self.mlp(x)  # [N, out_dim]
        x = self.bn(x)
        x = self.relu(x)
        return x.unsqueeze(2).unsqueeze(3)  # [N, out_dim, 1, 1]


# --- Topo (single-person, fixed layout) ---
class Topo(nn.Module):
    """Topological feature extraction (single-person, fixed variant)."""
    def __init__(self, dims=0, use_velocity=True):
        super(Topo, self).__init__()
        self.use_velocity = use_velocity
        self.vr = VietorisRipsComplex(dim=dims)
        self.pl = StructureElementLayer(n_elements=64)
        self.relu = nn.ReLU()

        if self.use_velocity:
            self.raw_alpha = nn.Parameter(torch.zeros(1))

    def L2_norm(self, weight):
        weight_norm = torch.norm(weight, 2, dim=1)
        return weight_norm

    def normalize(self, t):
        min_v = t.min(dim=-1, keepdim=True)[0].min(dim=-2, keepdim=True)[0]
        max_v = t.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
        return (t - min_v) / (max_v - min_v + 1e-6)

    def forward(self, x):
        # Input: [N, C, T, V] (single person; no multi-person M dim)
        if self.use_velocity:
            v = torch.zeros_like(x)
            v[:, :, 1:, :] = x[:, :, 1:, :] - x[:, :, :-1, :]

        pos_diff = x.unsqueeze(-1) - x.unsqueeze(-2)
        pos_diff = pos_diff.mean(-3)
        pos_struct = self.L2_norm(pos_diff)
        pos_struct = self.normalize(pos_struct)

        if self.use_velocity:
            vel_diff = v.unsqueeze(-1) - v.unsqueeze(-2)
            vel_diff = vel_diff.mean(-3)
            vel_struct = self.L2_norm(vel_diff)
            vel_struct = self.normalize(vel_struct)
            alpha = torch.sigmoid(self.raw_alpha)
            x_fused = (1 - alpha) * pos_struct + alpha * vel_struct
        else:
            x_fused = pos_struct

        x_fused = self.normalize(x_fused)
        x_out = self.vr(x_fused)
        x_out = make_tensor(x_out)
        x_out = self.pl(x_out)

        return x_out  # [N, 64]


class GraphMamba(nn.Module):
    """
    GraphMamba for Parkinson's disease scale prediction.

    Compact 4-layer variant:
    - Single-person setting (num_person=1)
    - Fixed input layout [B, T, V, C], e.g. [B, 240, 24, 3]
    - Reduced width/depth for small datasets
    """
    def __init__(self,
                 # Parkinson task
                 num_class=4,
                 num_point=24,
                 num_frame=240,
                 in_channels=3,
                 
                 # Graph structure
                 graph='graph.hybrik.Graph',
                 graph_args=dict(),
                 
                 # Model hyperparameters
                 drop_out=0,
                 adaptive=True,
                 alpha=False,
                 
                 # Contrastive learning
                 cl_mode=None,
                 multi_cl_weights=[1.0, 1.0],
                 cl_version='V0',
                 
                 # Legacy kwargs
                 num_person=1,
                 window_size=None,
                 joint_label=None,
                 # ========== HHT / Signal Dynamics Injection ==========
                 use_hht_injection=False,
                 hht_in_channels=10,
                 **kwargs):
        super(GraphMamba, self).__init__()

        if window_size is not None:
            num_frame = window_size

        self.num_class = num_class
        self.num_point = num_point
        self.num_frame = num_frame
        self.num_person = 1  # fixed single person
        self.cl_mode = cl_mode
        self.multi_cl_weights = multi_cl_weights
        self.cl_version = cl_version
        self.use_hht_injection = use_hht_injection
        self.hht_in_channels = hht_in_channels

        # Graph
        if graph is None:
            raise ValueError("Graph must be specified")
        Graph = import_class(graph)
        self.graph = Graph(**graph_args)
        A = self.graph.A

        print(f"[INFO] GraphMamba: {num_point} joints, {num_frame} frames, {num_class} classes (Single Person, 4 Layers)")
        if cl_mode:
            print(f"[INFO] Contrastive Learning: {cl_mode}, weights={multi_cl_weights}")
        if use_hht_injection:
            print(f"[INFO] HHT Signal Dynamics Injection: enabled (in_channels={hht_in_channels})")

        # Embedding + batch norm on joints
        self.data_bn = nn.BatchNorm1d(64 * num_point)
        self.to_joint_embedding = nn.Linear(in_channels, 64)
        self.pos_embedding = nn.Parameter(torch.randn(1, num_point, 64))

        # Topological branch
        self.topo = Topo()
        self.t0 = TopoTrans(out_dim=64)   # project topo to 64-d
        self.t1 = TopoTrans(out_dim=64)
        self.t2 = TopoTrans(out_dim=128)
        self.t3 = TopoTrans(out_dim=128)

        # GCN-Mamba stack (4 layers)
        self.l1 = TCN_GCN_unit_mamba(64, 64, A, adaptive=adaptive, alpha=alpha)
        self.l2 = TCN_GCN_unit_mamba(64, 128, A, adaptive=adaptive, alpha=alpha)
        self.l3 = TCN_GCN_unit_mamba(128, 128, A, adaptive=adaptive, alpha=alpha)
        self.l4 = TCN_GCN_unit_mamba(128, 128, A, adaptive=adaptive, alpha=alpha)

        if self.use_hht_injection:
            self.hht_proj_64 = nn.Linear(self.hht_in_channels, 64)
            self.hht_proj_128 = nn.Linear(self.hht_in_channels, 128)
        else:
            self.hht_proj_64 = None
            self.hht_proj_128 = None

        # Classifier head
        self.fc = nn.Linear(128, num_class)
        nn.init.normal_(self.fc.weight, 0, math.sqrt(2.0 / num_class))
        if self.fc.bias is not None:
            nn.init.constant_(self.fc.bias, 0)

        # ========== Dropout ==========
        if drop_out:
            self.drop_out = nn.Dropout(drop_out)
        else:
            self.drop_out = lambda x: x

        # Contrastive heads (channel dims aligned to backbone)
        if self.cl_mode == "ST-Multi-Level":
            # feat_mid has 128 channels (not 256)
            self.ren_mid = ST_RenovateNet(
                n_channel=128,  # was base_c*2 in full model
                n_frame=self.num_frame,
                n_joint=self.num_point,
                n_person=1,
                h_channel=256,  # smaller hidden dim
                n_class=self.num_class,
                version=self.cl_version,
            )

            self.ren_fin = ST_RenovateNet(
                n_channel=128,  # was base_c*2 in full model
                n_frame=self.num_frame,
                n_joint=self.num_point,
                n_person=1,
                h_channel=256,
                n_class=self.num_class,
                version=self.cl_version,
            )
            print(f"[INFO] ST-Multi-Level contrastive learning modules initialized (Lightweight)")
        else:
            self.ren_mid = self.ren_fin = None

        bn_init(self.data_bn, 1)

    def _hht_to_ctv(self, hht_feat: torch.Tensor, out_channels: int) -> torch.Tensor:
        """
        Map (B, T, F_hht) to (B, C, T, V); broadcast over joints and add to ST features.
        """
        if out_channels == 64:
            z = self.hht_proj_64(hht_feat)
        else:
            z = self.hht_proj_128(hht_feat)
        # z: [N, T, C] -> [N, C, T, V]
        z = z.transpose(1, 2).unsqueeze(-1).expand(-1, -1, -1, self.num_point).contiguous()
        return z

    def get_ST_Multi_Level_cl_output(self, x, feat_mid, feat_fin, label):
        """Contrastive outputs (feat_low / feat_high paths removed)."""
        logits = self.fc(x)
        
        N, C_mid, T, V = feat_mid.size()
        feat_mid_5d = feat_mid.unsqueeze(1).view(N * 1, C_mid, T, V)
        
        N, C_fin, T, V = feat_fin.size()
        feat_fin_5d = feat_fin.unsqueeze(1).view(N * 1, C_fin, T, V)

        cl_mid = self.ren_mid(feat_mid_5d, label.detach(), logits.detach())
        cl_fin = self.ren_fin(feat_fin_5d, label.detach(), logits.detach())
        
        cl_loss = (
            cl_mid * self.multi_cl_weights[0]
            + cl_fin * self.multi_cl_weights[1]
        )
        return logits, cl_loss

    def forward(self, x, hht_feat=None, y=None, get_cl_loss=False, return_rep=False):
        """
        Forward pass
        
        Args:
            x: Input [B, T, V, C] (e.g. T=150/240, V=21/24, C=3)
            hht_feat: optional [B, T, F_hht] HHT prior, time-aligned with x; required if use_hht_injection=True
            y: Labels (for contrastive learning)
            get_cl_loss: Whether to compute contrastive loss
            return_rep: Whether to return feature representation
        
        Returns:
            - If get_cl_loss=True: (logits, cl_loss)
            - If return_rep=True: features [B, 128]
            - Otherwise: logits [B, num_class]
        """
        # Layout: [B, T, V, C] -> [B, C, T, V]
        N, T, V, C = x.size()
        assert T == self.num_frame and V == self.num_point and C == 3, \
            f"Expected input shape [B, {self.num_frame}, {self.num_point}, 3], got {x.shape}"

        if self.use_hht_injection:
            if hht_feat is None:
                raise ValueError("use_hht_injection=True but hht_feat is None")
            if hht_feat.dim() != 3 or hht_feat.size(0) != N or hht_feat.size(1) != T:
                raise ValueError(
                    f"hht_feat expected [B={N}, T={T}, F], got {tuple(hht_feat.shape)}"
                )
            if hht_feat.size(-1) != self.hht_in_channels:
                raise ValueError(
                    f"hht_feat last dim must be hht_in_channels={self.hht_in_channels}, got {hht_feat.size(-1)}"
                )
        
        x = x.permute(0, 3, 1, 2).contiguous()  # [B, 3, 240, 24]

        # Topological features
        a = self.topo(x)  # [N, 64]

        # ========== Joint Embedding ==========
        x = rearrange(x, "n c t v -> (n t) v c").contiguous()  # [N*T, V, C]
        x = self.to_joint_embedding(x)  # [N*T, V, 64]
        x += self.pos_embedding[:, :self.num_point]
        x = rearrange(x, "(n t) v c -> n (v c) t", n=N, t=T).contiguous()  # [N, V*64, T]

        # ========== Batch Normalization ==========
        x = self.data_bn(x)
        x = x.view(N, V, 64, T).permute(0, 2, 3, 1).contiguous()  # [N, 64, T, V]

        # ========== GCN-Mamba Layers with Feature Extraction ==========
        h64 = self._hht_to_ctv(hht_feat, 64) if self.use_hht_injection else None
        h128 = self._hht_to_ctv(hht_feat, 128) if self.use_hht_injection else None

        x = self.l1(x + self.t0(a), hht_ctv=h64)  # [N, 64, T, V]
        x = self.l2(x + self.t1(a), hht_ctv=h128)  # [N, 128, T, V]
        feat_mid = x.clone()
        x = self.l3(x + self.t2(a), hht_ctv=h128)  # [N, 128, T, V]
        x = self.l4(x + self.t3(a), hht_ctv=h128)  # [N, 128, T, V]
        feat_fin = x.clone()

        # ========== Global Pooling ==========
        x = x.mean(dim=[2, 3])  # [N, 128]

        # ========== Dropout ==========
        x = self.drop_out(x)

        # Return representation only
        if return_rep:
            return x

        # Contrastive loss branch
        if get_cl_loss and self.cl_mode == "ST-Multi-Level" and y is not None:
            logits, cl_loss = self.get_ST_Multi_Level_cl_output(
                x, feat_mid, feat_fin, y  # no feat_low / feat_high
            )
            return logits, cl_loss

        # Default: classification logits
        logits = self.fc(x)
        return logits


# Backward-compatible alias
Model = GraphMamba