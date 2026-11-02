"""Parameter and operation counts"""
import csv
import json
import math
from pathlib import Path
import sys

import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'baselines_meta_d'))
import mmformer
import mmformer_custom
from ported_uhved.model import PortedHeteroModal


class DenseCounter:
    """Forward-shape counts"""
    def __init__(self, model):
        self.model = model
        self.macs = {}
        self.active_parameters = set()
        self.handles = []

    def __enter__(self):
        for name, module in self.model.named_modules():
            def hook(mod, args, output, name=name):
                self.active_parameters.update(id(p) for p in mod.parameters(recurse=False))
                count = 0
                if isinstance(mod, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                    count = output.numel() * (mod.in_channels // mod.groups) * math.prod(mod.kernel_size)
                elif isinstance(mod, nn.Linear):
                    count = output.numel() * mod.in_features
                elif mod.__class__.__name__ == 'SelfAttention':
                    batch, tokens, channels = output.shape
                    count = 2 * batch * tokens * tokens * channels
                elif mod.__class__.__name__ in ('ExplicitModalityCrossAttention', 'MetadataKVRouter'):
                    attention = mod.saved_attn_weights
                    batch, heads, queries, keys = attention.shape
                    count = 2 * batch * heads * queries * keys * mod.head_dim
                elif mod.__class__.__name__ == 'MetadataOnlyKV':
                    batch, queries, keys = mod.last_attention.shape
                    count = 2 * batch * queries * keys * mod.keys.shape[-1]
                self.macs[name] = self.macs.get(name, 0) + int(count)
            self.handles.append(module.register_forward_hook(hook))
        return self

    def __exit__(self, *exc):
        for handle in self.handles:
            handle.remove()

    def within(self, names):
        return sum(count for path, count in self.macs.items()
                   if any(path == name or path.startswith(name + '.') for name in names))


def parameter_count(modules):
    return sum(p.numel() for p in {id(p): p for m in modules for p in m.parameters()}.values())


def profile_model(model, image, present, label):
    """Model and component counts"""
    model.eval()
    if hasattr(model, 'is_training'):
        model.is_training = False
    with DenseCounter(model) as counter, torch.no_grad():
        output = model(image, present)
    total = parameter_count([model])


    active = sum(p.numel() for p in model.parameters() if id(p) in counter.active_parameters)
    row = {'model': label, 'parameters': total, 'inference_parameters': active,
           'macs': sum(counter.macs.values())}
    row.update(params_M=total/1e6, inference_params_M=active/1e6,
               GMACs=row['macs']/1e9, GFLOPs=2*row['macs']/1e9)
    if isinstance(output, dict):
        segmentation = output['seg']
        assert output['recon'] is None, 'Reconstruction must be disabled for inference profiling'
    else:
        segmentation = output[0] if isinstance(output, tuple) else output
    if segmentation.shape != (1, 4, 64, 64, 64) or not torch.isfinite(segmentation).all():
        raise RuntimeError('Invalid segmentation output for ' + label)
    components = []

    def component(label, names, note=''):
        modules = [model.get_submodule(name) for name in names]
        params, macs = parameter_count(modules), counter.within(names)
        components.append({'model': row['model'], 'component': label,
                           'parameters': params, 'params_M': params/1e6,
                           'macs': macs, 'GMACs': macs/1e9, 'GFLOPs': 2*macs/1e9, 'note': note})

    if hasattr(model, 'multimodal_transformer'):
        first = 'meta_cross_attn' if hasattr(model, 'meta_cross_attn') else 'flair_transformer'
        intra = [first, 't1ce_transformer', 't1_transformer', 't2_transformer']
        component('One intra-modal block (FLAIR)', [first], 'Excludes external positional embeddings and 1x1 token projections')
        component('Four intra-modal blocks', intra)
        component('Inter-modal transformer (unchanged)', ['multimodal_transformer'])
        component('All five transformer blocks', intra + ['multimodal_transformer'])
    else:


        names = ['routers'] if model.routers is not None else []
        if model.spatial_routers is not None:
            names = ['spatial_routers']
        component('Added metadata routers (four scales)', names,
                  'Baseline has no routers; native Gaussian fusion is excluded, not zero-cost')
        if names:
            component('Deepest metadata router', [names[0] + '.3'])
    return row, components


def run_complexity(output_dir=None, device=None):
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    torch.set_num_threads(min(4, torch.get_num_threads()))
    torch.manual_seed(0)
    image = torch.randn(1, 4, 64, 64, 64, device=device)
    present = torch.ones(1, 4, dtype=torch.bool, device=device)
    factories = [
        ('mmFormer', lambda: mmformer.Model(num_cls=4)),
        ('mmFormer + Meta-D', lambda: mmformer_custom.Model(num_cls=4)),
        ('U-HVED', lambda: PortedHeteroModal('uhved', False)),
        ('U-HVED + Meta-D', lambda: PortedHeteroModal('uhved', True, 'legacy')),
    ]
    rows, components = [], []
    for label, factory in factories:
        print('Complexity: ' + label, flush=True)
        model = factory().to(device)
        row, blocks = profile_model(model, image, present, label)
        rows.append(row)
        components.extend(blocks)
        del model
        if device.startswith('cuda'):
            torch.cuda.empty_cache()
    lines = [
        'Complexity: batch=1, crop=64^3, FP32, all four modalities present',
        'U-HVED: shared-latent, segmentation only',
        '1 MAC = 2 FLOPs. Dense Conv/Linear + QK^T and AV only.',
        'Excluded: bias adds, norms, activations, pooling/interpolation, softmax,',
        'elementwise operations (including U-HVED Gaussian fusion) and training losses.',
        '',
        '{:<24} {:>12} {:>14} {:>10} {:>10}'.format('Model', 'All params M', 'Infer params M', 'GMACs', 'GFLOPs'),
    ]
    for row in rows:
        lines.append('{model:<24} {params_M:>12.4f} {inference_params_M:>14.4f} {GMACs:>10.6f} {GFLOPs:>10.6f}'.format(**row))
    lines += ['', 'All params: registered tensors; Infer params: invoked modules',
              'Parameters: positional embeddings and modality dictionaries included',
              '', '{:<24} {:<40} {:>10} {:>10} {:>10}'.format('Model', 'Component', 'Params M', 'GMACs', 'GFLOPs')]
    for row in components:
        lines.append('{model:<24} {component:<40} {params_M:>10.6f} {GMACs:>10.6f} {GFLOPs:>10.6f}'.format(**row))
    for i in (0, 2):
        base, meta = rows[i:i+2]
        lines.append('{} -> Meta-D: all params {:+.4f} M; inference params {:+.4f} M; dense GFLOPs {:+.6f}'.format(
            base['model'], meta['params_M']-base['params_M'],
            meta['inference_params_M']-base['inference_params_M'], meta['GFLOPs']-base['GFLOPs']))
    report = '\n'.join(lines) + '\n'
    print('\n' + report, flush=True)
    out = Path(output_dir) if output_dir else ROOT / 'mmformer/results_complexity'
    out.mkdir(parents=True, exist_ok=True)
    for filename, data in [('model_complexity.csv', rows), ('bottleneck_complexity.csv', components)]:
        with (out / filename).open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(data[0]))
            writer.writeheader()
            writer.writerows(data)
    (out / 'complexity.txt').write_text(report)
    (out / 'complexity.json').write_text(json.dumps({
        'torch_version': torch.__version__, 'device': device, 'crop': [64,64,64],
        'batch_size': 1, 'modality_mask': [True]*4, 'flops_per_mac': 2,
        'counted_operations': ['Conv', 'Linear', 'attention QK^T', 'attention AV'],
        'models': rows, 'components': components}, indent=2))
    print('Saved: ' + str(out), flush=True)
    return rows, components


if __name__ == '__main__':
    run_complexity(sys.argv[1] if len(sys.argv) > 1 else None)
