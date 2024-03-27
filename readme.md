# [SFCFusion: Spatial–Frequency Collaborative Infrared and Visible Image Fusion](https://ieeexplore.ieee.org/document/10445750)

PyTorch implementation of SFCFusion, from the following paper:
[SFCFusion: Spatial–Frequency Collaborative Infrared and Visible Image Fusion](https://ieeexplore.ieee.org/document/10445750)
---
## News:
- [2024.3.27] Release source code.
- [2024.3.7] Paper published on IEEE Transactions on Instrumentation and Measurement. 
- [2023.8.6] Release Inference code for infrared-visible image feep fusion.
## Installation:
To test on a local machine, you may try
```python

```
## Evaluation
Our env is:
```
Python: 3.10.12
Pytorch: 1.13
```

Evaluation step:
1. Run main.m in MATLAB.
2. Run SFCFusionDeepfuse\main.py in python.
3. Edit the parameter deep in nsst_fuse.m from 0 to 1.
4. Run main.m in MATLAB. The final output is in fused folder.
## Training
Please use the train.py in SFCFusionDeepfuse folder.

## Acknowledgement
This work is inspired by [Densefuse](https://github.com/hli1221/imagefusion_densefuse)

## Citation
If you find this repository helpful, please consider citing:
```
@ARTICLE{10445750,
  author={Chen, Hanrui and Deng, Lei and Chen, Zhixiang and Liu, Chenhua and Zhu, Lianqing and Dong, Mingli and Lu, Xitian and Guo, Chentong},
  journal={IEEE Transactions on Instrumentation and Measurement}, 
  title={SFCFusion: Spatial–Frequency Collaborative Infrared and Visible Image Fusion}, 
  year={2024},
  volume={73},
  number={},
  pages={1-15},
  keywords={Frequency-domain analysis;Image fusion;Feature extraction;Image reconstruction;Fuses;Collaboration;Semantics;Deep learning;image fusion;multiscale transformation (MST);spatial-frequency;visible-infrared image},
  doi={10.1109/TIM.2024.3370752}}

```
## License
This project is released under the Apache License 2.0.

