# Temporal Misalignment in Multichannel Time Series

Research code accompanying the paper. The experiments impose temporal delays on
one sensor group while keeping the remaining channels fixed. They measure the
resulting classifier accuracy, compute the paper's label-free MDL alignment
cost, and compare MDL correction with SyncNet.

## Setup

Install the Python dependencies in an environment with PyTorch/CUDA as needed:

```bash
pip install -r requirements.txt
```

MONSTER datasets are downloaded through Hugging Face on first use. The two
SyncSine datasets are generated deterministically from their configurations.

## Reproduce all results

```bash
scripts/run_experiments.sh
```

The runner executes the full workflow from scratch:

1. Train CNN, ResNet, late fusion, Magnitude CNN, and SyncNet on every dataset.
2. Evaluate classifier accuracy and correction over the configured lag grids.
3. Compute unconditional and class-conditional MDL curves.
4. Evaluate SyncNet distances on the training folds.
5. Audit every complete real-world dataset, materialize its global MDL
   correction, and retrain the four classifiers at zero additional lag.
6. Generate every paper figure, table, and supporting CSV.

## Repository structure

```text
configs/                 one YAML experiment configuration per dataset
src/data.py              dataset loading, window cropping, and lag application
src/metric.py            MDL alignment cost and dataset-level evaluation
src/models.py            classifier and SyncNet architectures
src/train.py             model training and lag/correction evaluation
scripts/train_models.py  train models and write accuracy/correction curves
scripts/compute_metric.py
                         compute unconditional and conditional MDL curves
scripts/evaluate_syncnet_baselines.py
                         evaluate trained SyncNet models on training folds
scripts/prepare_global_corrected_datasets.py
                         materialize whole-dataset MDL corrections
scripts/plot_paper_results.py
                         generate all paper plots, tables, and analyses
scripts/run_experiments.sh
                         complete workflow
```

## Results and generated files

All numerical outputs use one of three result roots:

```text
results/original/          deployment-lag experiment and SyncNet comparison
results/global_audit/      MDL computed once on each complete real dataset
results/global_corrected/  classifiers retrained on globally corrected data
```

`results/original/checkpoints/` contains model checkpoints. Classifier,
correction, and SyncNet curves are CSV files directly below
`results/original/`. MDL output is separated only by whether labels were used:

```text
results/original/unconditional/{group_summary.csv,lag_curves.csv}
results/original/conditional/{group_summary.csv,lag_curves.csv}
```

The materialized global correction and its manifest are stored under
`data/global_corrected/`. Plotting never changes numerical results; it reads
the three result roots and writes figures and tables below
`paper/generated/{original,syncnet,conditional,global_audit}/`.

## Individual stages

Each stage can also be run independently:

```bash
PYTHONPATH=. python scripts/train_models.py \
  --configs configs/pamap2.yml \
  --results-dir results/original \
  --retrain

PYTHONPATH=. python scripts/compute_metric.py \
  --configs configs/*.yml \
  --results-dir results/original

PYTHONPATH=. python scripts/evaluate_syncnet_baselines.py \
  --configs configs/*.yml \
  --results-dir results/original

PYTHONPATH=. python scripts/plot_paper_results.py
```

`train_models.py` trains the classifiers first, selects the sensor group with
the largest average worst-lag accuracy loss, and trains SyncNet on that group.
Existing checkpoints are reused unless `--retrain` is supplied.

`compute_metric.py` always writes both definitions. The main-paper definition
is unconditional and label-free. The appendix definition supplies labels and
therefore fits/codes each class separately. Both use the same PCA
representation and raw bit scale.

## MDL API and interpretation

The direct API is:

```python
from src.metric import mdl_alignment_loss

result = mdl_alignment_loss(X, lags=[0, 5, 10], groups=[{3, 4, 5}])
conditional_result = mdl_alignment_loss(
    X, y, lags=[0, 5, 10], groups=[{3, 4, 5}]
)
```

Omitting `y` computes the unconditional main-paper metric. Supplying `y`
computes the class-conditional appendix variant; there is no separate mode
flag. PCA is the only representation.

For group `G`, `gain_bits` is the saving obtained by encoding the group through
the complement's PCA representation rather than directly. The reported
alignment cost is

```text
alignment_cost_bits(tau) = max_tau gain_bits(tau) - gain_bits(tau)
```

It is an unnormalized codelength in bits, is non-negative by construction, and
must be minimized. Positive experimental lags delay only the selected group.
All non-circular experiments crop every candidate to a common valid window, so
padding cannot influence the comparison.

## Configuration

There is exactly one `configs/<dataset>.yml` per dataset. It contains the data
source or synthetic generator, sensor groups, non-negative lag grid, model
list, and training parameters.
