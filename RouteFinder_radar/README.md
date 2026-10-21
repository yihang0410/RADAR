# RADAR on Multi-Task Asymmetric VRPs (RouteFinder)

RADAR trained jointly on 16 asymmetric VRP variants (N = 100): CVRP plus every combination of open routes (O),
backhauls (B), distance limits (L) and time windows (TW). The code is
[RouteFinder](https://github.com/ai4co/routefinder)'s (MIT license) adapted to asymmetric distance matrices,
with its encoder replaced by RADAR's (SVD initialization + Sinkhorn-normalized attention).

## Setup

```bash
cd RouteFinder_radar
pip install -r requirements.txt
```

For Blackwell GPUs install the CUDA 12.8 build of PyTorch
(`pip install torch==2.7.0 --index-url https://download.pytorch.org/whl/cu128`).
The bundled `rl4co/` is used, so do not install the `rl4co` package.

Download the test and validation sets ([Google Drive](https://drive.google.com/file/d/11eYr-td8NMJFEohBKxok-Aqgey-nTrGd/view?usp=sharing)) and unzip them here, giving
`data/<variant>/test/100.npz` (1000 instances per variant) and `data/<variant>/val/100.npz`.

## Test with our checkpoint

```bash
python test_radar.py
```

This evaluates `checkpoints/radar_mtvrp100.ckpt` on the 16 test sets (greedy, multi-start) and prints the
average cost and inference time per variant; the average over the 16 variants is 2.4842.

## Train

```bash
python run.py
```

The configuration is `configs/experiment/radar-mtvrp100.yaml` (300 epochs × 100k instances generated on the fly,
batch size 256). Checkpoints and CSV logs are written to `logs/radar-mtvrp100/`.
