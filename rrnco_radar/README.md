# RADAR on Real-World Routing (RRNCO Benchmark)

RADAR on the real-world ATSP, ACVRP and ACVRPTW benchmark of
[RRNCO](https://github.com/ai4co/real-routing-nco). The code is RRNCO's (MIT license) with its
encoder replaced by RADAR's.

## Setup

```bash
cd rrnco_radar
pip install -e .
python scripts/download_hf.py --no-models    # RRNCO data and test sets (~2.2 GB)
```

## Test with our checkpoints

```bash
python test.py --problem atsp    --datasets data/atsp/atsp_n100_seed3333_in_distribution.npz       --batch_size 32 --checkpoint checkpoints/radar/atsp/epoch_199.ckpt
python test.py --problem rcvrp   --datasets data/rcvrp/rcvrp_n100_seed3333_in_distribution.npz     --batch_size 32 --checkpoint checkpoints/radar/rcvrp/epoch_199.ckpt
python test.py --problem rcvrptw --datasets data/rcvrptw/rcvrptw_n100_seed3333_in_distribution.npz --batch_size 32 --checkpoint checkpoints/radar/rcvrptw/epoch_199.ckpt
```

Replace `in_distribution` with `out_of_distribution` or `in_distribution_cluster` for the other
test sets, and add `--no_aug` to disable the ×8 augmentation.

## Train

```bash
python train.py experiment=rradar_coords_mm env=atsp    # or env=rcvrp, env=rcvrptw
```

Add `logger=csv` to log locally instead of to Weights & Biases.
