"""Synthetic model tests"""
import argparse
from itertools import product
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT/'mmformer'), str(ROOT/'baselines_meta_d')]
import mmformer
import mmformer_custom
from ported_uhved.model import PortedHeteroModal


def attention(device):
    block = mmformer_custom.MMFormerCrossAttentionBlock().to(device).eval()
    tokens = torch.randn(1, 64, 512, device=device)
    patterns = [bits for bits in product((False, True), repeat=4) if any(bits)]
    with torch.no_grad():
        for bits in patterns:
            present = torch.tensor([bits], dtype=torch.bool, device=device)
            output = block(tokens, missing_mask=~present)
            weights = block.saved_attn_weights
            assert output.shape == tokens.shape and torch.isfinite(output).all()
            assert weights.shape == (1, 8, 64, 4)
            assert torch.allclose(weights.sum(-1), torch.ones_like(weights.sum(-1)), atol=1e-6)
            absent = weights.masked_select((~present)[:, None, None, :].expand_as(weights))
            assert not absent.numel() or torch.count_nonzero(absent).item() == 0

    present = torch.tensor([[False, True, True, False]], device=device)
    tokens = tokens.detach().requires_grad_(True)
    output = block(tokens, missing_mask=~present)
    output.square().mean().backward()
    for name, parameter in [('Q', block.cross_attn.q_proj.weight),
                            ('K', block.cross_attn.k_proj.weight),
                            ('V', block.cross_attn.v_proj.weight)]:
        assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum().item() > 0, name
    print('[PASS] attention: 15 masks, zero masked probability, Q/K/V gradients')
    print('       tokens [1,64,512] -> [1,64,512]; attention [1,8,64,4]')


def feature_fusion(device):
    image = torch.randn(1, 4, 64, 64, 64, device=device)
    present = torch.tensor([[False, True, True, True]], device=device)
    for name, factory in [('mmFormer', mmformer.Model), ('Meta-D FF', mmformer_custom.Model)]:
        model = factory(num_cls=4).to(device).eval()
        model.is_training = False
        with torch.no_grad():
            output = model(image, present)
        assert output.shape == (1, 4, 64, 64, 64) and torch.isfinite(output).all()
        assert torch.allclose(output.sum(1), torch.ones_like(output[:, 0]), atol=1e-5)
        print('[PASS] {}: [1,4,64,64,64] -> class probabilities [1,4,64,64,64]'.format(name))
        del model, output


def shared_latent(device):
    image = torch.randn(1, 4, 16, 16, 16, device=device)
    present = torch.tensor([[False, True, True, True]], device=device)
    for enabled in (False, True):
        model = PortedHeteroModal('uhved', enabled, 'legacy').to(device).eval()
        with torch.no_grad():
            output = model(image, present)
            altered = image.clone()
            altered[:, 0] += 50
            other = model(altered, present)['seg']
        assert output['seg'].shape == (1, 4, 16, 16, 16)
        assert output['recon'] is None and torch.isfinite(output['seg']).all()
        assert torch.allclose(output['seg'], other, atol=1e-5)
        if enabled:
            for router in model.routers:

                assert router.last_attention.shape == (1, 4, 4)
                assert torch.count_nonzero(router.last_attention[:, :, 3]).item() == 0
                assert torch.count_nonzero(router.last_weights[:, 3]).item() == 0
        print('[PASS] {}: logits [1,4,16,16,16], absent-image invariance'.format(
            'Meta-D SL' if enabled else 'U-HVED'))
        del model, output, other


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--component', choices=('attention','ff','sl','all'), default='all')
    parser.add_argument('--device', choices=('cpu','cuda'), default='cpu')
    args = parser.parse_args()
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; omit --device cuda to run on CPU')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(7)
    print('Synthetic verification | PyTorch {} | {}'.format(torch.__version__, args.device), flush=True)
    for name, check in [('attention', attention), ('ff', feature_fusion), ('sl', shared_latent)]:
        if args.component in ('all', name):
            check(args.device)
    print('Completed. Random weights and synthetic inputs; no Dice is measured.')


if __name__ == '__main__':
    main()
