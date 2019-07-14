# disease-network

### Abstract

This repository aims at:

- constructing a disease network which represents the relationship between different diseases
- constructing a temporal progression model of different diseases' progression over time for each individual patient
- disease prediction of a specific patient

via **Context-Sensitive Hawkes Process** or **cHawk**: multivariate Hawkes Process + every patient's context

![](http://latex.codecogs.com/gif.latex?\lambda_{d}^{i}(t)=\boldsymbol{\mu}_{d}^{\top} \boldsymbol{f}_{j}^{i}+\sum \alpha_{d, d_{j}^{i}} g\left(t-t_{j}^{i}\right) )

based on [Multiparameter Intelligent Monitoring in Intensive Care II (MIMIC II) clinical database](<https://www.physionet.org/mimic2/>)

### Reference

[Constructing disease network and temporal progression model via context-sensitive hawkes process](https://www.cc.gatech.edu/~lsong/papers/ChoDuCheSonSun15.pdf)

