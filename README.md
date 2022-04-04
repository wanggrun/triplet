# Solving Inefficiency of Self-supervised Representation Learning


This is the code of our paper "Solving Inefficiency of Self-supervised Representation Learning" (https://arxiv.org/abs/2104.08760).

[Guangrun Wang](https://wanggrun.github.io), [Keze Wang](https://kezewang.com/tutorials.html), [Guangcong Wang](https://wanggcong.github.io), [Philip H.S. Torr](https://www.robots.ox.ac.uk/~phst/), and [Liang Lin](http://www.linliang.net/)

See median_triplet_head.py for the median triplet loss. More updates are coming soon.


# About the paper

[Oral representation slices](https://drive.google.com/file/d/1lgEUss4UJS2HN2uuCYeFeY_E9cmF2G1y/view)

[Poster](https://drive.google.com/file/d/1gHC2yr9vQjNBAaZChvU64ORXqzvoVG9_/view)

[Leaderboard on SYSU-30k](https://paperswithcode.com/sota/person-re-identification-on-sysu-30k)


# Experiments

## ImageNet

An example of SSL training script on ImageNet:


```shell
bash tools/dist_train.sh configs/selfsup/triplet/r50_bs4096_accumulate4_ep1000_fp16_triplet_gpu12g.py 8
```

## COCO 2017

For object detection and instance segmentation tasks on COCO 2017, please go to the "triplet/benchmarks/detection/" folder and run the relevant scripts.


An example of training script on COCO 2017:


```shell
cd benchmarks/detection/
python convert-pretrain-to-detectron2.py  xxxxxx/release_ep800.pth  xxxxxx/release_detection_ep800.pkl
bash run.sh  configs/coco_R_50_C4_2x_moco.yaml   xxxxxx/release_detection_ep800.pkl
```



## PASCAL VOC

For PASCAL VOC07+12 Object Detection, please go to the "triplet/benchmarks/detection/" folder and run the relevant scripts.

## SYSU-30k


We next verify the effectiveness of our method on a more extensive data set, [SYSU-30k](https://github.com/wanggrun/SYSU-30k), that is 30 times larger than ImageNet both in terms of category number and image number.

Currently, SYSU-30k supports both [Google drive](https://drive.google.com/drive/folders/1MTxZ4UN_mbxjByZgcAki-H10zDzzeyuJ) collection and [Baidu Pan](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ#list/path=%2F) (code: 1qzv) collection.


# Installation

This repo has been tested in the following environment. More precisely, this repo is a modification on the OpenSelfSup. Installation and preparation follow that repo. Please acknowledge the great work of the team of OpenSelfSup.

For object detection and instance segmentation tasks, this repo follows OpenSelfSup and uses Detectron2. Thanks for their outstanding contributions.

Pytorch1.9

[OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup)


[detectron2](https://github.com/facebookresearch/detectron2)
