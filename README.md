# Meta-D

[Paper](https://mpsych.org/meta-d)

Meta-D models for the feature-fusion (mmFormer) and shared-latent (U-HVED) settings.
Includes synthetic checks and parameter/FLOP profiling.

## Usage

```bash
conda env create -f environment.yml
conda activate metad
python -m pip install -r requirements.txt

python examples/test_models.py --component all
PYTHONPATH=baselines_meta_d python -m ported_uhved.selftest
python mmformer/model_complexity.py
```
