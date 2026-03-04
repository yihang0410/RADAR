# RADAR: Learning to Route with Asymmetry-aware DistAnce Representations

**Hang Yi**, **Ziwei Huang**, **Yining Ma**, **Zhiguang Cao**

Paper: https://openreview.net/forum?id=lWdxX5s9T1

---

## Overview

RADAR is a neural combinatorial optimization framework designed for solving asymmetric routing problems, such as the Asymmetric Traveling Salesman Problem (ATSP). It enhances neural VRP solvers with the ability to effectively handle asymmetric distance matrices. RADAR leverages Singular Value Decomposition (SVD) to initialize compact embeddings that capture static asymmetry, and introduces Sinkhorn normalization to model dynamic asymmetry during attention interactions. Extensive experiments on synthetic and real-world benchmarks demonstrate strong generalization and superior performance across various asymmetric VRPs.

---

## Framework

<p align="center">
<img src="figures/framework.png" width="600">
</p>
