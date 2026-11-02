"""Modality routing"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MetadataKVRouter(nn.Module):
    """Modality K/V routing"""
    def __init__(self, embed_dim, num_heads=8, dropout=0.1, ffn_expansion=4):
        super().__init__()
        if embed_dim % num_heads:
            raise ValueError('embed_dim must be divisible by num_heads')
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.modality_encoding = nn.Parameter(torch.randn(1, 4, embed_dim))
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.norm_q = nn.LayerNorm(embed_dim)
        self.norm_ffn = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * ffn_expansion), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(embed_dim * ffn_expansion, embed_dim), nn.Dropout(dropout),
        )
        self.saved_attn_weights = None
        self.saved_attn_probabilities = None

    def forward(self, tokens, present_mask, return_details=False):

        if present_mask.ndim != 2 or present_mask.shape[1] != 4:
            raise ValueError('present_mask must have shape [B, 4]')
        if not torch.all(present_mask.any(dim=1)):
            raise ValueError('at least one modality must be present for every subject')
        b, n, _ = tokens.shape
        q = self.q_proj(self.norm_q(tokens)).view(b, n, self.num_heads, self.head_dim).transpose(1, 2)
        dictionary = self.modality_encoding.expand(b, -1, -1)
        k = self.k_proj(dictionary).view(b, 4, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(dictionary).view(b, 4, self.num_heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        missing = ~present_mask.to(dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(missing[:, None, None, :], float('-inf'))
        probabilities = F.softmax(logits, dim=-1)
        self.saved_attn_probabilities = probabilities.detach()
        attention = self.dropout(probabilities)
        self.saved_attn_weights = attention.detach()
        context = torch.matmul(attention, v).transpose(1, 2).contiguous().view(b, n, self.embed_dim)
        context = self.out_proj(context)
        routed = tokens + context
        routed = routed + self.ffn(self.norm_ffn(routed))
        if return_details:
            return routed, probabilities, context
        return routed
