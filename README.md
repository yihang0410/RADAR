# RADAR: Learning to Route with Asymmetry-aware Distance Representations

**Hang Yi**, **Ziwei Huang**, **Yining Ma**, **Zhiguang Cao**

📄 Paper: [RADAR](https://openreview.net/forum?id=lWdxX5s9T1)

---

## Overview

RADAR is a neural combinatorial optimization framework designed for solving asymmetric routing problems, such as the **Asymmetric Traveling Salesman Problem (ATSP)**.

The proposed framework enhances neural routing solvers with the ability to effectively model asymmetric distance matrices. RADAR leverages **Singular Value Decomposition (SVD)** to initialize compact embeddings that capture static asymmetry, and introduces **Sinkhorn normalization** to model dynamic asymmetry during attention interactions.

Extensive experiments on both synthetic and real-world benchmarks demonstrate strong generalization ability and superior performance across various asymmetric VRPs.

---

## Framework

<p align="center">
  <img src="image/framework.png" width="750">
</p>

---

## Dataset and Checkpoints

The datasets and pretrained checkpoints used in our experiments can be downloaded from the following links.

### Datasets

- **ATSP dataset:**  
https://drive.google.com/file/d/1dwKlConq9AhOObcTu-57wOsxr47FEhHU/view?usp=sharing

- **ACVRP dataset:**  
https://drive.google.com/file/d/1OdzFHqj_kvaSgHMRO0l4nvEuyr7RV2fK/view?usp=sharing


### Pretrained Models

- **ATSP checkpoint:**  
https://drive.google.com/file/d/1vO98NyK3DAaDBAJa5Y6bzWLfyM8QGs_0/view?usp=sharing

- **ACVRP checkpoint:**  
https://drive.google.com/file/d/10GFNnGh8pKHZbA-YqkhJj3YdaCEIpzic/view?usp=sharing


After downloading the files, please unzip them and place them into the corresponding folders.

For example:

```
RADAR
│
├── atsp
│   ├── dataset
│   │   └── (ATSP dataset files)
│   │
│   └── result
│       └── radar_official_checkpoint
```

---

## Training and Testing

### ATSP Example

Below we provide an example for training and evaluating RADAR on the **ATSP task**.

First, navigate to the `atsp` directory:

```bash
cd atsp
```

### Training

To train the model:

```bash
python train.py
```

### Testing

Before testing, modify the parameter `problem_cnt` in `test.py` to select the dataset size.

Supported problem sizes include:

- `problem_cnt = 100`
- `problem_cnt = 200`
- `problem_cnt = 500`
- `problem_cnt = 1000`

After setting the desired problem size, run:

```bash
python test.py
```

---

## Citation

If you find this work or code useful in your research, please consider citing our paper:

```bibtex
@inproceedings{yi2026radar,
  title={RADAR: Learning to Route with Asymmetry-aware Distance Representations},
  author={Yi, Hang and Huang, Ziwei and Cao, Zhiguang and Ma, Yining},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```
