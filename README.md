# SmoothRankingSVM

This repository contains the official implementation for the following paper on optimizing AUC by using a prototype learning approach:

* [Smooth Ranking SVM via Cutting-Plane Method](https://arxiv.org/abs/2401.14388)


This paper proposes a prototype learning approach based on cutting-plane method to optimize AUC. By penalizing the weight change in each iteration, the dramatic changes in the weight vector are prevented, and a smoother learning is achieved. Unlike typical machine learning algorithms, our approach achieves regularization through the proposed column generation strategy, and it learns simpler models than its relative counterparts. We show the benefits of utilizing our approach over various benchmark algorithms by testing it on various publicly available datasets.

Please consider citing our paper as follows:

```
@article{ozcan2024smooth,
  title={Smooth Ranking SVM via Cutting-Plane Method},
  author={Ozcan, Erhan Can and G{\"o}rg{\"u}l{\"u}, Berk and Baydogan, Mustafa G and Paschalidis, Ioannis Ch},
  journal={arXiv preprint arXiv:2401.14388},
  year={2024}
}
``` 


