import torch
import torch.nn as nn
import torch.nn.functional as F
import math


# =========================================================================
# Part 0: Basic Utilities & WTConv Operator
# =========================================================================
class BlurPool(nn.Module):
    def __init__(self, channels, stride=2, filt_size=3):
        super(BlurPool, self).__init__()
        if filt_size == 3:
            filt = torch.tensor([1., 2., 1.])
        elif filt_size == 4:
            filt = torch.tensor([1., 3., 3., 1.])
        else:
            filt = torch.tensor([1., 1.])
        filt_2d = filt[:, None] * filt[None, :]
        filt_2d = filt_2d / filt_2d.sum()
        self.register_buffer('filt', filt_2d[None, None, :, :].repeat((channels, 1, 1, 1)))
        self.stride = stride
        self.pad = [filt_size // 2] * 4

    def forward(self, x):
        x = F.pad(x, self.pad, mode='reflect')
        return F.conv2d(x, self.filt, stride=self.stride, groups=x.shape[1])


class DropPath(nn.Module):
    def __init__(self, drop_prob=0., scale_by_keep=True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
        if keep_prob > 0.0 and self.scale_by_keep:
            random_tensor.div_(keep_prob)
        return x * random_tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return torch.nn.init.trunc_normal_(tensor, mean=mean, std=std, a=a, b=b)


# --- Haar Wavelet Transform (MPS-compatible) ---
class HaarWaveletTransform(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

    def dwt(self, x):
        # Pure tensor operations for MPS compatibility
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2] + x01[:, :, :, 1::2] + x02[:, :, :, 0::2] + x02[:, :, :, 1::2]
        x2 = x01[:, :, :, 0::2] + x01[:, :, :, 1::2] - x02[:, :, :, 0::2] - x02[:, :, :, 1::2]
        x3 = x01[:, :, :, 0::2] - x01[:, :, :, 1::2] + x02[:, :, :, 0::2] - x02[:, :, :, 1::2]
        x4 = x01[:, :, :, 0::2] - x01[:, :, :, 1::2] - x02[:, :, :, 0::2] + x02[:, :, :, 1::2]
        # Returns: [B, 4C, H/2, W/2]
        return torch.cat([x1, x2, x3, x4], dim=1)


class WTConv2d(nn.Module):
    """
    Wavelet Transform Convolution (WTConv).
    Replaces the standard local aggregation branch in LWGA with
    frequency-decomposed convolutions for enhanced crack edge representation.
    """
    def __init__(self, in_channels, out_channels, kernel_size=5, stride=1, bias=True):
        super(WTConv2d, self).__init__()
        self.in_channels = in_channels
        self.wt = HaarWaveletTransform(in_channels)
        self.stride = stride

        # Low-frequency (LL) branch
        self.conv_ll = nn.Conv2d(in_channels, out_channels, kernel_size,
                                 padding=kernel_size // 2, stride=1, groups=1)

        # High-frequency (LH, HL, HH) branch
        self.conv_high = nn.Conv2d(in_channels * 3, out_channels * 3, kernel_size,
                                   padding=kernel_size // 2, stride=1, groups=in_channels)

        # Fusion layer
        self.fusion = nn.Conv2d(out_channels * 4, out_channels, 1)

    def forward(self, x):
        x_dwt = self.wt.dwt(x)
        B, C4, H_2, W_2 = x_dwt.shape
        C = C4 // 4

        x_ll = x_dwt[:, :C, :, :]
        x_high = x_dwt[:, C:, :, :]

        out_ll = self.conv_ll(x_ll)
        out_high = self.conv_high(x_high)

        out_cat = torch.cat([out_ll, out_high], dim=1)
        out = self.fusion(out_cat)

        # Bilinear upsampling to restore original resolution (replaces IDWT)
        return F.interpolate(out, scale_factor=2 * self.stride, mode='bilinear', align_corners=True)


# =========================================================================
# Part 1: Multi-scale Spectral Injection (MSI)
# =========================================================================
class HaarWaveletDownsampling(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.register_buffer('aa', torch.tensor([[1., 1.], [1., 1.]]) / 4)
        self.register_buffer('ad', torch.tensor([[1., -1.], [1., -1.]]) / 4)
        self.register_buffer('da', torch.tensor([[1., 1.], [-1., -1.]]) / 4)
        self.register_buffer('dd', torch.tensor([[1., -1.], [-1., 1.]]) / 4)

    def forward(self, x):
        B, C, H, W = x.shape
        filters = torch.stack([self.aa, self.ad, self.da, self.dd], dim=0)
        filters = filters.unsqueeze(1)
        filters = filters.repeat(C, 1, 1, 1)
        out = F.conv2d(x, filters, stride=2, groups=C)
        return out


class WaveletInjection(nn.Module):
    """
    Multi-scale Spectral Injection (MSI) module.
    Injects wavelet-derived spectral priors from the raw input image
    into each decoder stage to compensate for high-frequency texture loss.
    """
    def __init__(self, out_c, level):
        super().__init__()
        self.level = level
        if level > 1:
            self.pre_pool = nn.AvgPool2d(kernel_size=2 ** (level - 1), stride=2 ** (level - 1))
        else:
            self.pre_pool = nn.Identity()
        self.dwt = HaarWaveletDownsampling(in_channels=3)
        self.conv = nn.Conv2d(12, out_c, 1)

    def forward(self, raw_img):
        x = self.pre_pool(raw_img)
        x = self.dwt(x)
        return self.conv(x)


# =========================================================================
# Part 2: LWGA-WT Encoder Backbone
# =========================================================================
class FineGrainedStem(nn.Module):
    """
    Fine-Grained Stem for preserving micro-crack details during early downsampling.
    Uses two consecutive 3x3 convolutions with stride 2 instead of aggressive patch embedding.
    """
    def __init__(self, in_chans, stem_dim, norm_layer):
        super().__init__()
        self.conv1 = nn.Conv2d(in_chans, stem_dim // 2, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm1 = norm_layer(stem_dim // 2)
        self.act1 = nn.GELU()
        self.conv2 = nn.Conv2d(stem_dim // 2, stem_dim, kernel_size=3, stride=2, padding=1, bias=False)
        self.norm2 = norm_layer(stem_dim)

    def forward(self, x):
        x = self.act1(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return x


class DRFD(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.dim = dim
        self.outdim = dim * 2
        self.conv = nn.Conv2d(dim, dim * 2, kernel_size=3, stride=1, padding=1, groups=dim)
        self.conv_c = nn.Conv2d(dim * 2, dim * 2, kernel_size=3, stride=2, padding=1, groups=dim * 2)
        self.act_c = act_layer()
        self.norm_c = norm_layer(dim * 2)
        self.max_m = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.norm_m = norm_layer(dim * 2)
        self.fusion = nn.Conv2d(dim * 4, self.outdim, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        max_feat = self.norm_m(self.max_m(x))
        conv_feat = self.norm_c(self.act_c(self.conv_c(x)))
        return self.fusion(torch.cat([conv_feat, max_feat], dim=1))


class PA(nn.Module):
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.p_conv = nn.Sequential(
            nn.Conv2d(dim, dim * 4, 1, bias=False),
            norm_layer(dim * 4),
            act_layer(),
            nn.Conv2d(dim * 4, dim, 1, bias=False)
        )
        self.gate_fn = nn.Sigmoid()

    def forward(self, x):
        return x * self.gate_fn(self.p_conv(x))


class LA(nn.Module):
    """
    Local Aggregation branch with WTConv replacing standard convolution
    for spectral-spatial feature extraction.
    """
    def __init__(self, dim, norm_layer, act_layer):
        super().__init__()
        self.conv = nn.Sequential(
            WTConv2d(dim, dim, kernel_size=5),
            norm_layer(dim),
            act_layer()
        )

    def forward(self, x):
        return self.conv(x)


class MRA(nn.Module):
    def __init__(self, channel, att_kernel, norm_layer):
        super().__init__()
        att_padding = att_kernel // 2
        self.gate_fn = nn.Sigmoid()
        self.channel = channel
        self.max_m1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.max_m2 = BlurPool(channel, stride=3)
        self.H_att1 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att1 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.H_att2 = nn.Conv2d(channel, channel, (att_kernel, 3), 1, (att_padding, 1), groups=channel, bias=False)
        self.V_att2 = nn.Conv2d(channel, channel, (3, att_kernel), 1, (1, att_padding), groups=channel, bias=False)
        self.norm = norm_layer(channel)

    def forward(self, x):
        x_tem = self.max_m2(self.max_m1(x))
        x_h1 = self.H_att1(x_tem)
        x_w1 = self.V_att1(x_tem)
        x_h2 = self.inv_h_transform(self.H_att2(self.h_transform(x_tem)))
        x_w2 = self.inv_v_transform(self.V_att2(self.v_transform(x_tem)))
        att = self.norm(x_h1 + x_w1 + x_h2 + x_w2)
        return x[:, :self.channel, :, :] * F.interpolate(
            self.gate_fn(att), size=(x.shape[-2], x.shape[-1]), mode='nearest')

    def h_transform(self, x):
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1])).reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        return x.contiguous().reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1)

    def inv_h_transform(self, x):
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2])).reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., 0: shape[-2]].contiguous()

    def v_transform(self, x):
        x = x.permute(0, 1, 3, 2).contiguous()
        shape = x.size()
        x = torch.nn.functional.pad(x, (0, shape[-1])).reshape(shape[0], shape[1], -1)[..., :-shape[-1]]
        return x.contiguous().reshape(shape[0], shape[1], shape[2], 2 * shape[3] - 1).permute(0, 1, 3, 2).contiguous()

    def inv_v_transform(self, x):
        x = x.permute(0, 1, 3, 2).contiguous()
        shape = x.size()
        x = x.reshape(shape[0], shape[1], -1).contiguous()
        x = torch.nn.functional.pad(x, (0, shape[-2])).reshape(shape[0], shape[1], shape[-2], 2 * shape[-2])
        return x[..., 0: shape[-2]].contiguous().permute(0, 1, 3, 2).contiguous()


class GA12(nn.Module):
    def __init__(self, dim, act_layer):
        super().__init__()
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.uppool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.proj_1 = nn.Conv2d(dim, dim, 1)
        self.activation = act_layer()
        self.conv0 = nn.Conv2d(dim, dim, 5, padding=2, groups=dim)
        self.conv_spatial = nn.Conv2d(dim, dim, 7, stride=1, padding=9, groups=dim, dilation=3)
        self.conv1 = nn.Conv2d(dim, dim // 2, 1)
        self.conv2 = nn.Conv2d(dim, dim // 2, 1)
        self.conv_squeeze = nn.Conv2d(2, 2, 7, padding=3)
        self.conv = nn.Conv2d(dim // 2, dim, 1)
        self.proj_2 = nn.Conv2d(dim, dim, 1)

    def forward(self, x):
        x_ = self.activation(self.proj_1(self.downpool(x)))
        attn = torch.cat([self.conv1(self.conv0(x_)), self.conv2(self.conv_spatial(self.conv0(x_)))], dim=1)
        agg = torch.cat([torch.mean(attn, dim=1, keepdim=True), torch.max(attn, dim=1, keepdim=True)[0]], dim=1)
        sig = self.conv_squeeze(agg).sigmoid()
        return self.uppool(self.proj_2(
            x_ * self.conv(attn[:, :attn.shape[1] // 2] * sig[:, 0:1] + attn[:, attn.shape[1] // 2:] * sig[:, 1:2])))


class D_GA(nn.Module):
    def __init__(self, dim, norm_layer):
        super().__init__()
        self.norm = norm_layer(dim)
        self.attn = GA(dim)
        self.downpool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.uppool = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

    def forward(self, x):
        return self.uppool(self.norm(self.attn(self.downpool(x))))


class GA(nn.Module):
    def __init__(self, dim, head_dim=4, num_heads=None, qkv_bias=False,
                 attn_drop=0., proj_drop=0., proj_bias=False):
        super().__init__()
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5
        self.num_heads = num_heads if num_heads else dim // head_dim
        self.attention_dim = self.num_heads * self.head_dim
        self.qkv = nn.Linear(dim, self.attention_dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(self.attention_dim, dim, bias=proj_bias)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).contiguous()
        N = H * W
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)
        attn = ((q @ k.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        x = (self.attn_drop(attn) @ v).transpose(1, 2).reshape(B, H, W, self.attention_dim)
        return self.proj_drop(self.proj(x)).permute(0, 3, 1, 2).contiguous()


class LWGA_Block(nn.Module):
    def __init__(self, dim, stage, att_kernel, mlp_ratio, drop_path, act_layer, norm_layer):
        super().__init__()
        self.stage = stage
        self.dim_split = dim // 4
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Conv2d(dim, mlp_hidden_dim, 1, bias=False),
            norm_layer(mlp_hidden_dim),
            act_layer(),
            nn.Conv2d(mlp_hidden_dim, dim, 1, bias=False)
        )
        self.PA = PA(self.dim_split, norm_layer, act_layer)
        self.LA = LA(self.dim_split, norm_layer, act_layer)
        self.MRA = MRA(self.dim_split, att_kernel, norm_layer)
        if stage == 2:
            self.GA3 = D_GA(self.dim_split, norm_layer)
        elif stage == 3:
            self.GA4 = GA(self.dim_split)
            self.norm = norm_layer(self.dim_split)
        else:
            self.GA12 = GA12(self.dim_split, act_layer)
            self.norm = norm_layer(self.dim_split)
        self.norm1 = norm_layer(dim)

    def forward(self, x):
        shortcut = x.clone()
        x1, x2, x3, x4 = torch.split(x, [self.dim_split] * 4, dim=1)
        x1 = x1 + self.PA(x1)
        x2 = self.LA(x2)
        x3 = self.MRA(x3)
        if self.stage == 2:
            x4 = x4 + self.GA3(x4)
        elif self.stage == 3:
            x4 = self.norm(x4 + self.GA4(x4))
        else:
            x4 = self.norm(x4 + self.GA12(x4))
        return shortcut + self.norm1(self.drop_path(self.mlp(torch.cat((x1, x2, x3, x4), 1))))


class BasicStage(nn.Module):
    def __init__(self, dim, stage, depth, att_kernel, mlp_ratio, drop_path, norm_layer, act_layer):
        super().__init__()
        self.blocks = nn.Sequential(
            *[LWGA_Block(dim, stage, att_kernel, mlp_ratio, drop_path[i], act_layer, norm_layer)
              for i in range(depth)])

    def forward(self, x):
        return self.blocks(x)


class LWGANet(nn.Module):
    def __init__(self, in_chans=3, stem_dim=32, depths=(1, 2, 4, 2),
                 att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d,
                 act_layer=nn.GELU, mlp_ratio=2., drop_path_rate=0.0):
        super().__init__()
        self.Stem = FineGrainedStem(in_chans, stem_dim, norm_layer)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        stages_list = []
        for i_stage in range(len(depths)):
            stages_list.append(
                BasicStage(int(stem_dim * 2 ** i_stage), i_stage, depths[i_stage],
                           att_kernel[i_stage], mlp_ratio,
                           dpr[sum(depths[:i_stage]):sum(depths[:i_stage + 1])],
                           norm_layer, act_layer))
            if i_stage < len(depths) - 1:
                stages_list.append(
                    DRFD(dim=int(stem_dim * 2 ** i_stage), norm_layer=norm_layer, act_layer=act_layer))
        self.stages = nn.Sequential(*stages_list)
        self.out_indices = [0, 2, 4, 6]
        for i_emb, i_layer in enumerate(self.out_indices):
            self.add_module(f'norm{i_layer}', norm_layer(int(stem_dim * 2 ** i_emb)))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0)
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.Stem(x)
        outs = []
        for idx, stage in enumerate(self.stages):
            x = stage(x)
            if idx in self.out_indices:
                outs.append(getattr(self, f'norm{idx}')(x))
        return outs


# =========================================================================
# Part 3: Decoder (Neural Kolmogorov Mixer + HyperUp)
# =========================================================================
class FastKANConv2d(nn.Module):
    """
    Neural Kolmogorov Mixer (NKM).
    Replaces standard bottleneck with learnable RBF-based nonlinear mappings
    inspired by Kolmogorov-Arnold Networks (KAN).
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, grid_size=8):
        super().__init__()
        self.grid_size = grid_size
        self.base_conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding)
        self.rbf_conv = nn.Conv2d(in_channels * grid_size, out_channels, kernel_size, padding=padding, groups=1)
        self.ln = nn.InstanceNorm2d(in_channels, affine=True)
        grid_range = [-2, 2]
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = torch.linspace(grid_range[0] - h, grid_range[1] + h, grid_size)
        self.register_buffer("grid", grid)
        nn.init.kaiming_uniform_(self.base_conv.weight, nonlinearity='linear')
        nn.init.kaiming_uniform_(self.rbf_conv.weight, nonlinearity='linear')

    def forward(self, x):
        base = F.silu(self.base_conv(x))
        x_norm = self.ln(x)
        x_rbf = torch.exp(
            -((x_norm.unsqueeze(2).contiguous() - self.grid.view(1, 1, -1, 1, 1)) / (2 / self.grid_size)) ** 2)
        return base + self.rbf_conv(x_rbf.contiguous().reshape(x.shape[0], -1, x.shape[2], x.shape[3]))


class ScharrEdge(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        kernel = torch.tensor([[-3., 0., 3.], [-10., 0., 10.], [-3., 0., 3.]])
        self.register_buffer('kx', kernel.view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.register_buffer('ky', kernel.t().view(1, 1, 3, 3).repeat(in_channels, 1, 1, 1))
        self.in_channels = in_channels

    def forward(self, x):
        return torch.sqrt(
            F.conv2d(x, self.kx, padding=1, groups=self.in_channels) ** 2 +
            F.conv2d(x, self.ky, padding=1, groups=self.in_channels) ** 2 + 1e-6)


class EGABlock(nn.Module):
    """Edge-Gated Attention (EGA) for skip feature refinement."""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )
        self.edge_extractor = ScharrEdge(in_c)
        self.edge_attention = nn.Sequential(nn.Conv2d(in_c, 1, 1), nn.Sigmoid())
        self.final = nn.Conv2d(out_c, out_c, 1)

    def forward(self, x):
        return self.final(self.conv(x) * (1 + self.edge_attention(self.edge_extractor(x))))


class MambaOutBlock(nn.Module):
    """
    Gated Local Mixer (GLM) inspired by MambaOut.
    Uses large-kernel depthwise convolutions for efficient long-range dependency modeling.
    """
    def __init__(self, dim, expansion=1.5, kernel_size=7):
        super().__init__()
        mid_dim = int(dim * expansion)
        self.proj_up = nn.Conv2d(dim, 2 * mid_dim, 1)
        self.dwconv = nn.Conv2d(mid_dim, mid_dim, kernel_size=kernel_size,
                                padding=kernel_size // 2, groups=mid_dim, bias=False)
        self.norm = nn.GroupNorm(num_groups=1, num_channels=mid_dim)
        self.act = nn.SiLU()
        self.proj_down = nn.Conv2d(mid_dim, dim, 1)

    def forward(self, x):
        shortcut = x
        x = self.proj_up(x)
        x_feat, x_gate = x.chunk(2, dim=1)
        x_feat = self.act(self.norm(self.dwconv(x_feat)))
        return self.proj_down(x_feat * F.silu(x_gate)) + shortcut


class HyperUpBlock(nn.Module):
    """
    Hyper-connected Upsampling (HyperUp) block.
    Fuses upsampled deep features, EGA-refined skip features,
    and MSI wavelet priors for topology-aware reconstruction.
    """
    def __init__(self, in_c, skip_c, raw_c, out_c):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.skip_ega = EGABlock(skip_c, skip_c)
        self.has_skip = skip_c > 0
        self.reduce = nn.Sequential(
            nn.Conv2d(in_c + skip_c + raw_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.SiLU()
        )
        self.mamba_mixer = MambaOutBlock(out_c, expansion=1.5, kernel_size=7)

    def forward(self, x, skip, raw_feat):
        x = self.up(x)
        if self.has_skip:
            if x.shape[2:] != skip.shape[2:]:
                x = F.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=True)
            out = torch.cat([x, self.skip_ega(skip), raw_feat], dim=1)
        else:
            if x.shape[2:] != raw_feat.shape[2:]:
                x = F.interpolate(x, size=raw_feat.shape[2:], mode='bilinear', align_corners=True)
            out = torch.cat([x, raw_feat], dim=1)
        return self.mamba_mixer(self.reduce(out))


# =========================================================================
# Part 4: Full StarKAN Model
# =========================================================================
class StarKAN(nn.Module):
    """
    StarKAN: A Lightweight Frequency-Aware Framework for Crack Segmentation
    in Geotechnical and Infrastructure Monitoring.

    Architecture:
        - Encoder: LWGA-WT backbone with Fine-Grained Stem and WTConv
        - Bottleneck: Neural Kolmogorov Mixer (NKM)
        - Decoder: HyperUp blocks with MSI and EGA
        - Loss: BCE-Dice + Soft-CLDice with deep supervision
    """
    def __init__(self, n_classes=1):
        super().__init__()
        self.backbone = LWGANet(
            in_chans=3, stem_dim=32, depths=(1, 2, 4, 2),
            att_kernel=(11, 11, 11, 11), norm_layer=nn.BatchNorm2d,
            act_layer=nn.GELU, mlp_ratio=2., drop_path_rate=0.0
        )
        c = [32, 64, 128, 256]
        self.kan_neck = FastKANConv2d(c[3], 256, kernel_size=1, padding=0, grid_size=8)

        self.inj4 = WaveletInjection(16, level=4)
        self.inj3 = WaveletInjection(16, level=3)
        self.inj2 = WaveletInjection(16, level=2)
        self.inj1 = WaveletInjection(16, level=1)

        self.up1 = HyperUpBlock(256, c[2], 16, 128)
        self.up2 = HyperUpBlock(128, c[1], 16, 64)
        self.up3 = HyperUpBlock(64, c[0], 16, 32)
        self.up4 = HyperUpBlock(32, 0, 16, 32)

        self.ds_head1 = nn.Conv2d(128, n_classes, 1)
        self.ds_head2 = nn.Conv2d(64, n_classes, 1)
        self.final_conv = nn.Conv2d(32, n_classes, 1)

    def forward(self, x):
        r4 = self.inj4(x)
        r3 = self.inj3(x)
        r2 = self.inj2(x)
        r1 = self.inj1(x)

        feats = self.backbone(x)
        c1, c2, c3, c4 = feats[0], feats[1], feats[2], feats[3]

        neck = self.kan_neck(c4)

        d1 = self.up1(neck, c3, r4)
        d2 = self.up2(d1, c2, r3)
        d3 = self.up3(d2, c1, r2)
        d4 = self.up4(d3, None, r1)

        final = F.interpolate(self.final_conv(d4), scale_factor=2, mode='bilinear', align_corners=True)

        if self.training:
            o1 = F.interpolate(self.ds_head1(d1), size=x.shape[2:], mode='bilinear')
            o2 = F.interpolate(self.ds_head2(d2), size=x.shape[2:], mode='bilinear')
            return final, o1, o2
        return final


if __name__ == '__main__':
    model = StarKAN()
    x = torch.randn(2, 3, 320, 320)
    out = model(x)
    total = sum(p.numel() for p in model.parameters())
    print(f"StarKAN Total Params: {total / 1e6:.2f}M")
