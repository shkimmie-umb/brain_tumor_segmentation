"""Fusion tests"""
import torch

from .model import MetadataOnlyKV, PortedHeteroModal, niftynet_seg_loss, product_of_experts


def main():
    torch.manual_seed(7)
    present = torch.tensor([[True, False, True, False]])
    x = torch.randn(1, 4, 16, 16, 16)
    labels = torch.nn.functional.one_hot(torch.randint(0, 4, (1, 16, 16, 16)), 4)
    labels = labels.permute(0, 4, 1, 2, 3).float()

    router = MetadataOnlyKV(8)
    features = torch.randn(1, 4, 8, 2, 2, 2)
    weights = router(features, present)
    assert torch.equal(weights[:, (1, 3)], torch.zeros_like(weights[:, (1, 3)]))
    assert torch.equal(router.last_attention[:, :, (1, 3)],
                       torch.zeros_like(router.last_attention[:, :, (1, 3)]))
    weights[:, 0].sum().backward()
    for name, parameter in (('query', router.query.weight), ('keys', router.keys),
                             ('values', router.values)):
        assert parameter.grad is not None and parameter.grad.abs().sum() > 0, name

    mu = torch.randn(1, 4, 2, 2, 2, 2)
    logvar = torch.zeros_like(mu)
    mean, lv = product_of_experts(mu, logvar, present)
    assert torch.allclose(mean, (mu[:, 0] + mu[:, 2]) / 3, atol=1e-6)
    assert torch.allclose(lv, torch.full_like(lv, -torch.log(torch.tensor(3.0)).item()), atol=1e-6)

    for family in ('hemis', 'uhved'):
        for metad in (False, True):
            model = PortedHeteroModal(family, metad).eval()
            with torch.no_grad():
                first = model(x, present)['seg']
                altered = x.clone()
                altered[:, (1, 3)] += 100
                second = model(altered, present)['seg']
            assert first.shape == (1, 4, 16, 16, 16)
            assert torch.allclose(first, second, atol=1e-5), (family, metad)
            assert torch.isfinite(niftynet_seg_loss(first, labels))
            clone = PortedHeteroModal(family, metad).eval()
            clone.load_state_dict(model.state_dict(), strict=True)
            with torch.no_grad():
                assert torch.equal(first, clone(x, present)['seg'])
            print(f'{family} metad={metad}: shapes, masking, checkpoint reload OK', flush=True)
        base = PortedHeteroModal(family, False).eval()
        augmented = PortedHeteroModal(family, True).eval()
        augmented.load_state_dict(base.state_dict(), strict=False)
        for router in augmented.routers:
            torch.nn.init.zeros_(router.keys)
            torch.nn.init.zeros_(router.weight_head.weight)
            torch.nn.init.zeros_(router.weight_head.bias)
        with torch.no_grad():
            baseline_prediction = base(x, present)['seg']
            neutral_prediction = augmented(x, present)['seg']
        assert torch.allclose(baseline_prediction, neutral_prediction, atol=1e-5), family
        print(f'{family}: neutral router reproduces baseline fusion', flush=True)
    print('Metadata-only K/V gradients and Gaussian product OK', flush=True)


if __name__ == '__main__':
    main()
