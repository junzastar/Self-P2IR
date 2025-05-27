# [TMI'24] Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection

<p align="center">
<img src="figures/pipeline.png" alt="intro" width="100%"/>
</p>

Official Implementation of "[Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection](https://arxiv.org/abs/2504.15152)"

[Jun Zhou](https://scholar.google.com/citations?hl=zh-CN&user=pEgMa-UAAAAJ), Bingchen Gao, Kai Wang, [Jialun Pei](https://scholar.google.com/citations?user=1lPivLsAAAAJ&hl=en), [Pheng-Ann Heng](https://scholar.google.com/citations?user=OFdytjoAAAAJ&hl=zh-CN) and [Jing Qin](https://harry-qinjing.github.io/)

## Installation
**Conda virtual environment**

We recommend using conda to setup the environment.

If you have already installed conda, please use the following commands.

```bash
conda create -n SelfTraining python=3.8
conda activate SelfTraining
pip install -r requirements.txt
```

## Prepare 
Download the processed P2I-LReg dataset, you can download it [here](https://github.com/junzastar/Self-P2IR/edit/main/README.md). Our Pretrained model can be downloaded [here](https://github.com/junzastar/Self-P2IR/edit/main/README.md)


## Train
```bash
sh train.sh
```

## Citation
If you find the code useful, please cite our paper.
```latex
@article{zhou2025landmark,
  title={Landmark-Free Preoperative-to-Intraoperative Registration in Laparoscopic Liver Resection},
  author={Zhou, Jun and Gao, Bingchen and Wang, Kai and Pei, Jialun and Heng, Pheng-Ann and Qin, Jing},
  journal={arXiv preprint arXiv:2504.15152},
  year={2025}
}
```

## Acknowledgment
Our code is developed based on [Lepard](https://github.com/rabbityl/lepard) and [NDP](https://github.com/rabbityl/DeformationPyramid). We thank the authors for providing the source code.
