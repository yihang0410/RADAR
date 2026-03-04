# RADAR: Learning to Route with Asymmetry-aware DistAnce Representations

**Hang Yi**, **Ziwei Huang**, **Yining Ma**, **Zhiguang Cao**

Paper: https://openreview.net/forum?id=lWdxX5s9T1

---

## Overview

RADAR is a neural combinatorial optimization framework designed for solving **asymmetric routing problems**, such as the **Asymmetric Traveling Salesman Problem (ATSP)** and related routing tasks. A core current challenge is encoding the relational features in asymmetric distance matrices of VRPs. Early attempts directly encoded these matrices but often failed to produce compact embeddings and generalized poorly at scale. In this paper, we propose RADAR, a scalable neural framework that augments existing neural VRP solvers with the ability to handle asymmetric inputs. RADAR addresses asymmetry from both static and dynamic perspectives. It leverages Singular Value Decomposition (SVD) on the asymmetric distance matrix to initialize compact and generalizable embeddings that inherently encode the static asymmetry in the inbound and outbound costs of each node. To further model dynamic asymmetry in embedding interactions during encoding, it replaces the standard softmax with Sinkhorn normalization that imposes joint row and column distance awareness in attention weights. Extensive experiments on synthetic and real-world benchmarks across various VRPs show that RADAR outperforms strong baselines on both in-distribution and out-of-distribution instances, demonstrating robust generalization and superior performance in solving asymmetric VRPs.

---
