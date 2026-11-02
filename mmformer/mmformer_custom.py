import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_
import torch
import math
from layers import general_conv3d_prenorm, fusion_prenorm

basic_dims = 8
transformer_basic_dims = 512
mlp_dim = 4096
num_heads = 8
depth = 1
num_modals = 4
patch_size = 4


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.e1_c1 = nn.Conv3d(in_channels=1, out_channels=basic_dims, kernel_size=3, stride=1, padding=1, padding_mode='reflect', bias=True)
        self.e1_c2 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e1_c3 = general_conv3d_prenorm(basic_dims, basic_dims, pad_type='reflect')
        self.e2_c1 = general_conv3d_prenorm(basic_dims, basic_dims*2, stride=2, pad_type='reflect')
        self.e2_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e2_c3 = general_conv3d_prenorm(basic_dims*2, basic_dims*2, pad_type='reflect')
        self.e3_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims*4, stride=2, pad_type='reflect')
        self.e3_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e3_c3 = general_conv3d_prenorm(basic_dims*4, basic_dims*4, pad_type='reflect')
        self.e4_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*8, stride=2, pad_type='reflect')
        self.e4_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e4_c3 = general_conv3d_prenorm(basic_dims*8, basic_dims*8, pad_type='reflect')
        self.e5_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*16, stride=2, pad_type='reflect')
        self.e5_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')
        self.e5_c3 = general_conv3d_prenorm(basic_dims*16, basic_dims*16, pad_type='reflect')

    def forward(self, x):
        x1 = self.e1_c1(x)
        x1 = x1 + self.e1_c3(self.e1_c2(x1))
        x2 = self.e2_c1(x1)
        x2 = x2 + self.e2_c3(self.e2_c2(x2))
        x3 = self.e3_c1(x2)
        x3 = x3 + self.e3_c3(self.e3_c2(x3))
        x4 = self.e4_c1(x3)
        x4 = x4 + self.e4_c3(self.e4_c2(x4))
        x5 = self.e5_c1(x4)
        x5 = x5 + self.e5_c3(self.e5_c2(x5))
        return x1, x2, x3, x4, x5

class Decoder_sep(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_sep, self).__init__()
        self.d4 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')
        self.d3 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')
        self.d2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')
        self.d1 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.d4_c1(self.d4(x5))
        cat_x4 = torch.cat((de_x5, x4), dim=1)
        de_x4 = self.d4_out(self.d4_c2(cat_x4))
        de_x4 = self.d3_c1(self.d3(de_x4))
        cat_x3 = torch.cat((de_x4, x3), dim=1)
        de_x3 = self.d3_out(self.d3_c2(cat_x3))
        de_x3 = self.d2_c1(self.d2(de_x3))
        cat_x2 = torch.cat((de_x3, x2), dim=1)
        de_x2 = self.d2_out(self.d2_c2(cat_x2))
        de_x2 = self.d1_c1(self.d1(de_x2))
        cat_x1 = torch.cat((de_x2, x1), dim=1)
        de_x1 = self.d1_out(self.d1_c2(cat_x1))
        return self.softmax(self.seg_layer(de_x1))

class Decoder_fuse(nn.Module):
    def __init__(self, num_cls=4):
        super(Decoder_fuse, self).__init__()
        self.d4_c1 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_c2 = general_conv3d_prenorm(basic_dims*16, basic_dims*8, pad_type='reflect')
        self.d4_out = general_conv3d_prenorm(basic_dims*8, basic_dims*8, k_size=1, padding=0, pad_type='reflect')
        self.d3_c1 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_c2 = general_conv3d_prenorm(basic_dims*8, basic_dims*4, pad_type='reflect')
        self.d3_out = general_conv3d_prenorm(basic_dims*4, basic_dims*4, k_size=1, padding=0, pad_type='reflect')
        self.d2_c1 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_c2 = general_conv3d_prenorm(basic_dims*4, basic_dims*2, pad_type='reflect')
        self.d2_out = general_conv3d_prenorm(basic_dims*2, basic_dims*2, k_size=1, padding=0, pad_type='reflect')
        self.d1_c1 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_c2 = general_conv3d_prenorm(basic_dims*2, basic_dims, pad_type='reflect')
        self.d1_out = general_conv3d_prenorm(basic_dims, basic_dims, k_size=1, padding=0, pad_type='reflect')

        self.seg_d4 = nn.Conv3d(in_channels=basic_dims*16, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d3 = nn.Conv3d(in_channels=basic_dims*8, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d2 = nn.Conv3d(in_channels=basic_dims*4, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_d1 = nn.Conv3d(in_channels=basic_dims*2, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.seg_layer = nn.Conv3d(in_channels=basic_dims, out_channels=num_cls, kernel_size=1, stride=1, padding=0, bias=True)
        self.softmax = nn.Softmax(dim=1)

        self.up2 = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=4, mode='trilinear', align_corners=True)
        self.up8 = nn.Upsample(scale_factor=8, mode='trilinear', align_corners=True)
        self.up16 = nn.Upsample(scale_factor=16, mode='trilinear', align_corners=True)

        self.RFM5 = fusion_prenorm(in_channel=basic_dims*16, num_cls=num_cls)
        self.RFM4 = fusion_prenorm(in_channel=basic_dims*8, num_cls=num_cls)
        self.RFM3 = fusion_prenorm(in_channel=basic_dims*4, num_cls=num_cls)
        self.RFM2 = fusion_prenorm(in_channel=basic_dims*2, num_cls=num_cls)
        self.RFM1 = fusion_prenorm(in_channel=basic_dims*1, num_cls=num_cls)

    def forward(self, x1, x2, x3, x4, x5):
        de_x5 = self.RFM5(x5)
        pred4 = self.softmax(self.seg_d4(de_x5))
        de_x5 = self.d4_c1(self.up2(de_x5))
        de_x4 = self.RFM4(x4)
        de_x4 = torch.cat((de_x4, de_x5), dim=1)
        de_x4 = self.d4_out(self.d4_c2(de_x4))
        pred3 = self.softmax(self.seg_d3(de_x4))
        de_x4 = self.d3_c1(self.up2(de_x4))
        de_x3 = self.RFM3(x3)
        de_x3 = torch.cat((de_x3, de_x4), dim=1)
        de_x3 = self.d3_out(self.d3_c2(de_x3))
        pred2 = self.softmax(self.seg_d2(de_x3))
        de_x3 = self.d2_c1(self.up2(de_x3))
        de_x2 = self.RFM2(x2)
        de_x2 = torch.cat((de_x2, de_x3), dim=1)
        de_x2 = self.d2_out(self.d2_c2(de_x2))
        pred1 = self.softmax(self.seg_d1(de_x2))
        de_x2 = self.d1_c1(self.up2(de_x2))
        de_x1 = self.RFM1(x1)
        de_x1 = torch.cat((de_x1, de_x2), dim=1)
        de_x1 = self.d1_out(self.d1_c2(de_x1))
        logits = self.seg_layer(de_x1)
        pred = self.softmax(logits)
        return pred, (self.up2(pred1), self.up4(pred2), self.up8(pred3), self.up16(pred4))

class MaskModal(nn.Module):
    def __init__(self):
        super(MaskModal, self).__init__()
    def forward(self, x, mask):
        B, K, C, H, W, Z = x.size()
        y = torch.zeros_like(x)
        y[mask, ...] = x[mask, ...]
        return y.view(B, -1, H, W, Z)


class ExplicitModalityCrossAttention(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == self.embed_dim

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)


        self.scale = self.head_dim ** -0.5
        self.saved_attn_weights = None


        self.modality_encoding = nn.Parameter(torch.randn(1, 4, embed_dim))

    def forward(self, query, missing_mask=None, routing_mode="hard", soft_penalty=1.0):
        B, L_q, _ = query.size()


        Q = self.q_proj(query).view(B, L_q, self.num_heads, self.head_dim).transpose(1, 2)


        metadata_tokens = self.modality_encoding.expand(B, -1, -1)
        K = self.k_proj(metadata_tokens).view(B, 4, self.num_heads, self.head_dim).transpose(1, 2)


        V = self.v_proj(metadata_tokens).view(B, 4, self.num_heads, self.head_dim).transpose(1, 2)


        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale


        if missing_mask is not None and routing_mode == "hard":
            mask_expanded = missing_mask.unsqueeze(1).unsqueeze(1).bool()
            scores = scores.masked_fill(mask_expanded, float('-inf'))
        elif missing_mask is not None and routing_mode == "soft":
            mask_expanded = missing_mask.unsqueeze(1).unsqueeze(1).to(scores.dtype)
            scores = scores - float(soft_penalty) * mask_expanded
        elif routing_mode not in ("hard", "soft", "none"):
            raise ValueError("routing_mode must be hard, soft, or none")


        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        self.saved_attn_weights = attn.detach()

        context = torch.matmul(attn, V)
        context = context.transpose(1, 2).contiguous().view(B, L_q, self.embed_dim)
        return self.out_proj(context), attn


class MMFormerCrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim=512, num_heads=8, dropout=0.1, ffn_expansion=4):
        super().__init__()
        self.cross_attn = ExplicitModalityCrossAttention(embed_dim, num_heads, dropout)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)

        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion),
            GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_expansion, embed_dim),
            nn.Dropout(dropout)
        )

    @property
    def saved_attn_weights(self):
        return self.cross_attn.saved_attn_weights

    def forward(self, target_modality, missing_mask=None, routing_mode="hard", soft_penalty=1.0):
        q = self.norm_q(target_modality)

        attn_out, _ = self.cross_attn(query=q, missing_mask=missing_mask, routing_mode=routing_mode, soft_penalty=soft_penalty)

        x = target_modality + attn_out
        ffn_out = self.ffn(self.norm_ffn(x))
        return x + ffn_out


class SelfAttention(nn.Module):
    def __init__(self, dim, heads=8, qkv_bias=False, qk_scale=None, dropout_rate=0.0):
        super().__init__()
        self.num_heads = heads
        head_dim = dim // heads
        self.scale = qk_scale or head_dim ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(dropout_rate)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(dropout_rate)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

class PreNormDrop(nn.Module):
    def __init__(self, dim, dropout_rate, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.fn = fn
    def forward(self, x):
        return self.dropout(self.fn(self.norm(x)))

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x):
        return self.fn(self.norm(x))

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x):
        return self.fn(x) + x

class GELU(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.gelu(x)

class Transformer(nn.Module):
    def __init__(self, embedding_dim, depth, heads, mlp_dim, dropout_rate=0.1, n_levels=1, n_points=4):
        super(Transformer, self).__init__()
        self.cross_attention_list = nn.ModuleList([Residual(PreNormDrop(embedding_dim, dropout_rate, SelfAttention(embedding_dim, heads=heads, dropout_rate=dropout_rate))) for _ in range(depth)])
        self.cross_ffn_list = nn.ModuleList([Residual(PreNorm(embedding_dim, nn.Sequential(nn.Linear(embedding_dim, mlp_dim), GELU(), nn.Dropout(dropout_rate), nn.Linear(mlp_dim, embedding_dim), nn.Dropout(dropout_rate)))) for _ in range(depth)])
        self.depth = depth

    def forward(self, x, pos):
        for j in range(self.depth):
            x = x + pos
            x = self.cross_attention_list[j](x)
            x = self.cross_ffn_list[j](x)
        return x


class Model(nn.Module):
    def __init__(self, num_cls=4):
        super(Model, self).__init__()


        self.embed_dim = 512
        self.p_size = 4

        self.flair_encoder = Encoder()
        self.t1ce_encoder = Encoder()
        self.t1_encoder = Encoder()
        self.t2_encoder = Encoder()


        self.flair_encode_conv = nn.Conv3d(basic_dims*16, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.t1ce_encode_conv = nn.Conv3d(basic_dims*16, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.t1_encode_conv = nn.Conv3d(basic_dims*16, self.embed_dim, kernel_size=1, stride=1, padding=0)
        self.t2_encode_conv = nn.Conv3d(basic_dims*16, self.embed_dim, kernel_size=1, stride=1, padding=0)


        self.flair_pos = nn.Parameter(torch.zeros(1, self.p_size**3, self.embed_dim))
        self.t1ce_pos = nn.Parameter(torch.zeros(1, self.p_size**3, self.embed_dim))
        self.t1_pos = nn.Parameter(torch.zeros(1, self.p_size**3, self.embed_dim))
        self.t2_pos = nn.Parameter(torch.zeros(1, self.p_size**3, self.embed_dim))


        self.meta_cross_attn = MMFormerCrossAttentionBlock(embed_dim=self.embed_dim, num_heads=num_heads)
        self.t1ce_transformer = MMFormerCrossAttentionBlock(embed_dim=self.embed_dim, num_heads=num_heads)
        self.t1_transformer = MMFormerCrossAttentionBlock(embed_dim=self.embed_dim, num_heads=num_heads)
        self.t2_transformer = MMFormerCrossAttentionBlock(embed_dim=self.embed_dim, num_heads=num_heads)


        self.multimodal_transformer = Transformer(embedding_dim=self.embed_dim, depth=depth, heads=num_heads, mlp_dim=mlp_dim, n_levels=num_modals)
        self.multimodal_decode_conv = nn.Conv3d(self.embed_dim*num_modals, basic_dims*16*num_modals, kernel_size=1, padding=0)

        self.masker = MaskModal()
        self.decoder_fuse = Decoder_fuse(num_cls=num_cls)
        self.decoder_sep = Decoder_sep(num_cls=num_cls)
        self.is_training = False

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                torch.nn.init.kaiming_normal_(m.weight)

    def forward(self, x, mask, routing_mode=None, soft_penalty=1.0):
        B = x.size(0)


        flair_x1, flair_x2, flair_x3, flair_x4, flair_x5 = self.flair_encoder(x[:, 0:1, :, :, :])
        t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5 = self.t1ce_encoder(x[:, 1:2, :, :, :])
        t1_x1, t1_x2, t1_x3, t1_x4, t1_x5 = self.t1_encoder(x[:, 2:3, :, :, :])
        t2_x1, t2_x2, t2_x3, t2_x4, t2_x5 = self.t2_encoder(x[:, 3:4, :, :, :])


        flair_token = self.flair_encode_conv(flair_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim)
        t1ce_token = self.t1ce_encode_conv(t1ce_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim)
        t1_token = self.t1_encode_conv(t1_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim)
        t2_token = self.t2_encode_conv(t2_x5).permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim)


        missing_mask = (mask == False)
        routing_mode = routing_mode or getattr(self, "routing_mode", "hard")
        if (~missing_mask).sum(dim=1).min().item() == 0:
            raise ValueError("At least one modality must be present for T_max attention.")


        flair_q = flair_token + self.flair_pos
        flair_intra_token = self.meta_cross_attn(target_modality=flair_q, missing_mask=missing_mask, routing_mode=routing_mode, soft_penalty=soft_penalty)

        t1ce_q = t1ce_token + self.t1ce_pos
        t1ce_intra_token = self.t1ce_transformer(target_modality=t1ce_q, missing_mask=missing_mask, routing_mode=routing_mode, soft_penalty=soft_penalty)

        t1_q = t1_token + self.t1_pos
        t1_intra_token = self.t1_transformer(target_modality=t1_q, missing_mask=missing_mask, routing_mode=routing_mode, soft_penalty=soft_penalty)

        t2_q = t2_token + self.t2_pos
        t2_intra_token = self.t2_transformer(target_modality=t2_q, missing_mask=missing_mask, routing_mode=routing_mode, soft_penalty=soft_penalty)


        flair_intra_x5 = flair_intra_token.view(B, self.p_size, self.p_size, self.p_size, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()
        t1ce_intra_x5 = t1ce_intra_token.view(B, self.p_size, self.p_size, self.p_size, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()
        t1_intra_x5 = t1_intra_token.view(B, self.p_size, self.p_size, self.p_size, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()
        t2_intra_x5 = t2_intra_token.view(B, self.p_size, self.p_size, self.p_size, self.embed_dim).permute(0, 4, 1, 2, 3).contiguous()

        if self.is_training:
            flair_pred = self.decoder_sep(flair_x1, flair_x2, flair_x3, flair_x4, flair_x5)
            t1ce_pred = self.decoder_sep(t1ce_x1, t1ce_x2, t1ce_x3, t1ce_x4, t1ce_x5)
            t1_pred = self.decoder_sep(t1_x1, t1_x2, t1_x3, t1_x4, t1_x5)
            t2_pred = self.decoder_sep(t2_x1, t2_x2, t2_x3, t2_x4, t2_x5)


        x1 = self.masker(torch.stack((flair_x1, t1ce_x1, t1_x1, t2_x1), dim=1), mask)
        x2 = self.masker(torch.stack((flair_x2, t1ce_x2, t1_x2, t2_x2), dim=1), mask)
        x3 = self.masker(torch.stack((flair_x3, t1ce_x3, t1_x3, t2_x3), dim=1), mask)
        x4 = self.masker(torch.stack((flair_x4, t1ce_x4, t1_x4, t2_x4), dim=1), mask)
        x5_intra = self.masker(torch.stack((flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5), dim=1), mask)

        flair_intra_x5, t1ce_intra_x5, t1_intra_x5, t2_intra_x5 = torch.chunk(x5_intra, num_modals, dim=1)
        multimodal_token_x5 = torch.cat((flair_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim),
                                         t1ce_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim),
                                         t1_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim),
                                         t2_intra_x5.permute(0, 2, 3, 4, 1).contiguous().view(B, -1, self.embed_dim),
                                        ), dim=1)
        multimodal_pos = torch.cat((self.flair_pos, self.t1ce_pos, self.t1_pos, self.t2_pos), dim=1)

        multimodal_inter_token_x5 = self.multimodal_transformer(multimodal_token_x5, multimodal_pos)
        multimodal_inter_x5 = self.multimodal_decode_conv(multimodal_inter_token_x5.view(B, self.p_size, self.p_size, self.p_size, self.embed_dim*num_modals).permute(0, 4, 1, 2, 3).contiguous())

        fuse_pred, preds = self.decoder_fuse(x1, x2, x3, x4, multimodal_inter_x5)

        if self.is_training:
            return fuse_pred, (flair_pred, t1ce_pred, t1_pred, t2_pred), preds
        return fuse_pred
