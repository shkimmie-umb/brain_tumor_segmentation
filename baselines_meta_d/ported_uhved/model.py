"""HeMIS / U-HVED models"""
from itertools import combinations

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
from metad_routing import MetadataKVRouter


ORIGINAL_ORDER = (2, 1, 3, 0)


class PreActBlock(nn.Module):
    """Pre-activation convolutions"""
    def __init__(self, in_channels, middle_channels, out_channels):
        super().__init__()
        self.norm1 = nn.InstanceNorm3d(in_channels, eps=1e-6, affine=True)
        self.conv1 = nn.Conv3d(in_channels, middle_channels, 3, padding=1)
        self.norm2 = nn.InstanceNorm3d(middle_channels, eps=1e-6, affine=True)
        self.conv2 = nn.Conv3d(middle_channels, out_channels, 3, padding=1)

    def forward(self, x):
        x = self.conv1(F.leaky_relu(self.norm1(x), negative_slope=0.02))
        return self.conv2(F.leaky_relu(self.norm2(x), negative_slope=0.02))


class ModalityEncoder(nn.Module):
    def __init__(self, variational):
        super().__init__()
        self.first = nn.Sequential(nn.Conv3d(1, 8, 1), nn.LeakyReLU(0.02))
        self.blocks = nn.ModuleList([
            PreActBlock(8, 8, 8),
            PreActBlock(8, 16, 16),
            PreActBlock(16, 32, 32),
            PreActBlock(32, 64, 64),
        ])

    def forward(self, x):
        x = self.first(x)
        skips = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            skips.append(x)
            if i < 3:
                x = F.max_pool3d(x, 2)
        return skips


class MetadataOnlyKV(nn.Module):
    """Pooled Q; modality K/V"""
    def __init__(self, channels, hidden=32):
        super().__init__()
        self.query = nn.Linear(channels, hidden)
        self.keys = nn.Parameter(torch.randn(4, hidden) * 0.02)
        self.values = nn.Parameter(torch.randn(4, hidden) * 0.02)
        self.weight_head = nn.Linear(hidden * 2, 1)
        nn.init.normal_(self.weight_head.weight, std=0.02)
        nn.init.zeros_(self.weight_head.bias)
        self.last_attention = None
        self.last_weights = None

    def forward(self, features, present):

        if not present.any(dim=1).all():
            raise ValueError('At least one modality must be present per subject')
        q = self.query(features.mean(dim=(-3, -2, -1)))
        attention_logits = torch.matmul(q, self.keys.t()) * (q.shape[-1] ** -0.5)
        attention_logits = attention_logits.masked_fill(~present[:, None, :], -torch.inf)
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.matmul(attention, self.values)


        own_attention = attention.diagonal(dim1=1, dim2=2)
        logits = (self.weight_head(torch.cat((q, context), dim=-1)).squeeze(-1)
                  + torch.log(own_attention.clamp_min(1e-8)))
        logits = logits.masked_fill(~present, -torch.inf)
        weights = torch.softmax(logits, dim=-1)
        self.last_attention = attention.detach()
        self.last_weights = weights.detach()
        return weights


class SpatialMetadataResidual(nn.Module):
    """Spatial routing residual"""
    def __init__(self, channels):
        super().__init__()
        self.router = MetadataKVRouter(channels, num_heads=1 if channels < 16 else 4,
                                       dropout=0.0)
        self.gate = nn.Parameter(torch.tensor(0.01))

    def forward(self, feature, present):
        tokens = feature.flatten(2).transpose(1, 2)
        outputs = []

        for chunk in tokens.split(32768, dim=1):
            if self.training and torch.is_grad_enabled():
                routed = checkpoint(self.router, chunk, present, use_reentrant=False)
            else:
                routed = self.router(chunk, present)
            outputs.append(chunk + self.gate.tanh() * (routed - chunk))
        return torch.cat(outputs, dim=1).transpose(1, 2).reshape_as(feature)


class Decoder(nn.Module):
    def __init__(self, variational, output_channels):
        super().__init__()
        mult = 1 if variational else 4
        self.blocks = nn.ModuleList([
            PreActBlock(32 * mult + 16 * mult, 32, 16),
            PreActBlock(16 + 8 * mult, 16, 8),
            PreActBlock(8 + 4 * mult, 8, 4),
        ])
        self.final = nn.Conv3d(4, output_channels, 1)

    def forward(self, skips):
        x = skips[-1]
        for block, skip in zip(self.blocks, reversed(skips[:-1])):
            x = F.interpolate(x, size=skip.shape[-3:], mode='trilinear', align_corners=False)
            x = block(torch.cat((x, skip), dim=1))
        return self.final(x)


def product_of_experts(mu, logvar, present, scale=None):
    """Gaussian fusion"""
    present = present.to(dtype=mu.dtype)[..., None, None, None, None]
    precision = present / (torch.exp(logvar.clamp(-50, 50)) + 1e-7)
    if scale is not None:
        precision = precision * scale[..., None, None, None, None]
    total_precision = 1 + precision.sum(dim=1)
    posterior_mu = (mu * precision).sum(dim=1) / total_precision
    posterior_logvar = torch.log(total_precision.reciprocal() + 1e-7)
    return posterior_mu, posterior_logvar


class PortedHeteroModal(nn.Module):
    def __init__(self, family, metad=False, metad_variant='legacy'):
        super().__init__()
        if family not in ('hemis', 'uhved'):
            raise ValueError(f'Unknown family: {family}')
        self.family, self.metad = family, metad
        if metad_variant not in ('legacy', 'spatial'):
            raise ValueError(f'Unknown Meta-D variant: {metad_variant}')
        if metad_variant == 'spatial' and (family != 'uhved' or not metad):
            raise ValueError('spatial variant requires Meta-D U-HVED')
        self.metad_variant = metad_variant
        variational = family == 'uhved'
        self.encoders = nn.ModuleList([ModalityEncoder(variational) for _ in range(4)])
        self.seg_decoder = Decoder(variational, 4)
        self.reconstruction_decoders = (nn.ModuleList([Decoder(True, 1) for _ in range(4)])
                                        if variational else None)
        router_channels = (4, 8, 16, 32) if variational else (8, 16, 32, 64)
        self.routers = (nn.ModuleList([MetadataOnlyKV(c) for c in router_channels])
                        if metad and metad_variant == 'legacy' else None)
        self.spatial_routers = (nn.ModuleList([SpatialMetadataResidual(c) for c in router_channels])
                                if metad and metad_variant == 'spatial' else None)

    def forward(self, image, present, compute_recon=False):
        if present.ndim == 1:
            present = present[None].expand(image.shape[0], -1)
        if image.shape[1] != 4 or present.shape != (image.shape[0], 4):
            raise ValueError('Expected image [B,4,D,H,W] and mask [B,4]')
        image = image[:, ORIGINAL_ORDER]
        present = present[:, ORIGINAL_ORDER].bool()
        encoded = [encoder(image[:, i:i+1]) for i, encoder in enumerate(self.encoders)]
        skips, posteriors = [], []
        for level in range(4):
            features = torch.stack([encoded[i][level] for i in range(4)], dim=1)
            if self.family == 'hemis':
                weights = (self.routers[level](features, present) if self.metad else
                           present.float() / present.sum(1, keepdim=True))
                weights = weights[..., None, None, None, None]
                mean = (features * weights).sum(dim=1)
                variance = ((features - mean[:, None]).square() * weights).sum(dim=1)
                skips.append(torch.cat((mean, variance), dim=1))
            else:
                mu, logvar = features.chunk(2, dim=2)
                logvar = logvar.clamp(-50, 50)
                scale = None
                if self.metad and self.metad_variant == 'legacy':
                    weights = self.routers[level](mu, present)
                    scale = weights * present.sum(dim=1, keepdim=True)
                posterior_mu, posterior_logvar = product_of_experts(mu, logvar, present, scale)
                z = (posterior_mu if not self.training else posterior_mu +
                     torch.exp(0.5 * posterior_logvar) * torch.randn_like(posterior_mu))
                if self.spatial_routers is not None:
                    z = self.spatial_routers[level](z, present)
                skips.append(z)
                posteriors.append((mu, logvar))
        seg = self.seg_decoder(skips)
        recon = None
        if self.reconstruction_decoders is not None and (self.training or compute_recon):
            recon = torch.cat([decoder(skips) for decoder in self.reconstruction_decoders], dim=1)

        return {'seg': seg, 'recon': recon, 'posteriors': posteriors}


def niftynet_seg_loss(logits, target):
    """Cross-entropy + Dice"""
    labels = target.argmax(dim=1)
    cross_entropy = F.cross_entropy(logits, labels)
    probabilities = torch.softmax(logits, dim=1)
    dims = tuple(range(2, logits.ndim))
    numerator = 2 * (probabilities * target).sum(dim=dims) + 1e-5
    denominator = probabilities.square().sum(dim=dims) + target.sum(dim=dims) + 1e-5
    return cross_entropy + (1 - numerator / denominator).mean()


def uhved_kl_loss(posteriors, present):
    """Subset KL"""
    present = present[:, ORIGINAL_ORDER].bool()
    inter, prior = 0.0, 0.0
    for mu, logvar in posteriors:
        full_mu, full_lv = product_of_experts(mu, logvar, present)
        level_inter, level_prior = 0.0, 0.0
        for size in range(1, 5):
            for subset in combinations(range(4), size):
                subset_mask = torch.zeros_like(present)
                subset_mask[:, list(subset)] = True
                subset_mu, subset_lv = product_of_experts(mu, logvar, subset_mask)

                level_inter = level_inter + 0.5 * (-1 + subset_lv - full_lv +
                    (full_lv.exp() + (full_mu - subset_mu).square()) /
                    (subset_lv.exp() + 1e-7)).mean()
                level_prior = level_prior + 0.5 * (-1 - subset_lv +
                    subset_lv.exp() + subset_mu.square()).mean()
        inter = inter + level_inter / 14
        prior = prior + level_prior / 15
    return (inter + prior) / len(posteriors)
