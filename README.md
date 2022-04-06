# Solving Inefficiency of Self-supervised Representation Learning


This is the code of our paper "Solving Inefficiency of Self-supervised Representation Learning" (https://arxiv.org/abs/2104.08760).

[Guangrun Wang](https://wanggrun.github.io), [Keze Wang](https://kezewang.com/tutorials.html), [Guangcong Wang](https://wanggcong.github.io), [Philip H.S. Torr](https://www.robots.ox.ac.uk/~phst/), and [Liang Lin](http://www.linliang.net/)



# About the paper

[Oral representation slices](https://drive.google.com/file/d/1lgEUss4UJS2HN2uuCYeFeY_E9cmF2G1y/view)

[Poster](https://drive.google.com/file/d/1gHC2yr9vQjNBAaZChvU64ORXqzvoVG9_/view)

[Leaderboard on SYSU-30k](https://paperswithcode.com/sota/person-re-identification-on-sysu-30k)


# Experiments

## Our pretrained models

Just list a few pretrained models here:

| Model            | Top 1 Acc | Download                                                                          |
|:-----------------|:------------|:---------------------------------------------------------------------------------:|
| shorter epochs   | 73.8%     | [:arrow_down:](https://drive.google.com/file/d/1ZKXgFIrnREuX94l8ARQF5ID9gTwleWjr/view?usp=sharing) |
| longer epochs    | 75.9%     | [:arrow_down:](https://drive.google.com/file/d/19nO2IrT856-0N9BUyZlOrmwafYx-aHNe/view?usp=sharing) |



## ImageNet

An example of SSL training script on ImageNet:


```shell
bash tools/dist_train.sh configs/selfsup/triplet/r50_bs4096_accumulate4_ep1000_fp16_triplet_gpu3090 8
```

An example of linear evaluation script on ImageNet:

```shell
python tools/extract_backbone_weights.py  xxxxxxxxx/ssl_ep940.pth    xxxxxx/release_smooth_ep940.pth
bash benchmarks/dist_train_linear.sh configs/benchmarks/linear_classification/imagenet/r50_last_cos_for_MoreSslEpoch.py  xxxxxx/release_smooth_ep940.pth
```


<em>This repo can achieve a 73.8% top-1 accuracy for 200 epochs' SSL training and a 75.9% top-1 accuracy for 700-900 epochs' SSL training on Imagenet. </em>
  <table><thead><tr><th>Method</th><th>top-1 accuracy</th><th> epochs</th></tr></thead><tbody>
  	<tr><td>supervised</td><td>76.3</td><td>100</td></tr>
  	<tr><td>supervised</td><td>78.4</td><td>270</td></tr>
  	<tr><td>supervised + linear eval</td><td>74.1</td><td>100</td></tr>
  	<tr><td>Random</td><td>4.4</td><td>0</td></tr>
  	<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>38.8</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>47.0</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1807.05520" target="_blank" rel="noopener noreferrer">DeepCluster</a></td><td>46.9</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>56.6</td><td>200</td></tr>
  	<tr><td><a href="http://openaccess.thecvf.com/content_CVPR_2020/papers/Zhan_Online_Deep_Clustering_for_Unsupervised_Representation_Learning_CVPR_2020_paper.pdf" target="_blank" rel="noopener noreferrer">ODC</a></td><td>53.4</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>60.6</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>69.3</td><td>1000</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>61.9</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>67.0</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>71.1</td><td>800</td></tr><tr>
  	<td><a href="https://arxiv.org/abs/2006.09882" target="_blank" rel="noopener noreferrer">SwAV (single crop)</a></td><td>69.1</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2006.09882" target="_blank" rel="noopener noreferrer">SwAV (multi crop)</a></td><td>72.7</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL </a></td><td>71.5</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL </a></td><td>72.5</td><td>300</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL </a></td><td>74.3</td><td>1000</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2011.10566" target="_blank" rel="noopener noreferrer">SimSiam </a></td><td>68.1</td><td>100</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2011.10566" target="_blank" rel="noopener noreferrer">SimSiam </a></td><td>70.0</td><td>200</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2011.10566" target="_blank" rel="noopener noreferrer">SimSiam </a></td><td>70.8</td><td>400</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2011.10566" target="_blank" rel="noopener noreferrer">SimSiam </a></td><td>71.3</td><td>800</td></tr>
  	<tr><td>Triplet</td><td>73.8</td><td>200</td></tr>
  	<tr><td>Triplet</td><td>75.9</td><td>700-900</td></tr>
  </tbody></table>




## COCO 2017

For object detection and instance segmentation tasks on COCO 2017, please go to the "triplet/benchmarks/detection/" folder and run the relevant scripts.

Note: For the organizational structure of the COCO 2017 dataset and the installation of the operating environment, please check the official documentation of Detectron2.

An example of training script on COCO 2017:


```shell
cd benchmarks/detection/
python convert-pretrain-to-detectron2.py  xxxxxx/release_ep800.pth  xxxxxx/release_detection_ep800.pkl
bash run.sh  configs/coco_R_50_C4_2x_moco.yaml   xxxxxx/release_detection_ep800.pkl
```


<em>This repo can achieve a 41.7% AP(box) and a 36.2% AP(mask) on COCO 2017. </em>
  <table><thead><tr><th>Method</th><th>AP(box)</th><th> AP(mask)</th></tr></thead><tbody>
  	<tr><td>supervised</td><td>40.0</td><td>34.7</td></tr>
  	<tr><td>Random</td><td>35.6</td><td>31.4</td></tr>
  	<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>40.0</td><td>35.0</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>40.0</td><td>34.9</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>39.4</td><td>34.5</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>39.6</td><td>34.6</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>40.9</td><td>35.5</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>40.9</td><td>35.5</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL </a></td><td>40.3</td><td>35.1</td></tr>
  	<tr><td>Triplet</td><td>41.7</td><td>36.2</td></tr>
  </tbody></table>


## PASCAL VOC

For PASCAL VOC07+12 Object Detection, please go to the "triplet/benchmarks/detection/" folder and run the relevant scripts.

Note: For the organizational structure of the VOC07+12 dataset and the installation of the operating environment, please check the official documentation of Detectron2.

It is worth noting that because the VOC dataset is much smaller than the COCO 2017 dataset, multiple experiments should be performed on VOC, and the average of the results of the multiple experiments should be reported.


An example of training script on PASCAL VOC07+12:


```shell
cd benchmarks/detection/
python convert-pretrain-to-detectron2.py  xxxxxx/release_ep800.pth  xxxxxx/release_detection_ep800.pkl
bash run.sh  configs/pascal_voc_R_50_C4_24k_moco.yaml   xxxxxx/release_detection_ep800.pkl
```

<em>This repo can achieve a 82.6% AP50(box), a 56.9% AP(box), and a 63.8% AP75(box) on VOC07+12. </em>
  <table><thead><tr><th>Method</th><th>AP50</th><th> AP</th><th> AP75</th></tr></thead><tbody>
  	<tr><td>supervised</td><td>81.6</td><td>54.2</td><td>59.8</td></tr>
  	<tr><td>Random</td><td>59.0</td><td>32.8</td><td>31.6</td></tr>
  	<tr><td><a href="https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf" target="_blank" rel="noopener noreferrer">Relative-Loc</a></td><td>80.4</td><td>55.1</td><td>61.2</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1803.07728" target="_blank" rel="noopener noreferrer">Rotation-Pred</a></td><td>80.9</td><td>55.5</td><td>61.4</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1805.01978" target="_blank" rel="noopener noreferrer">NPID</a></td><td>80.0</td><td>54.1</td><td>59.5</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2002.05709" target="_blank" rel="noopener noreferrer">SimCLR</a></td><td>79.4</td><td>51.5</td><td> 55.6</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/1911.05722" target="_blank" rel="noopener noreferrer">MoCo</a></td><td>81.4</td><td>56.0</td><td>62.2</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2003.04297" target="_blank" rel="noopener noreferrer">MoCo v2</a></td><td>82.0</td><td>56.6</td><td>62.9</td></tr>
  	<tr><td><a href="https://arxiv.org/abs/2006.07733" target="_blank" rel="noopener noreferrer">BYOL </a></td><td>81.0</td><td>51.9</td><td>56.5</td></tr>
  	<tr><td>Triplet</td><td>82.6</td><td>56.9</td><td>63.8</td></tr>
  </tbody></table>


## SYSU-30k


We next verify the effectiveness of our method on a more extensive data set, [SYSU-30k](https://github.com/wanggrun/SYSU-30k), that is 30 times larger than ImageNet both in terms of category number and image number.

Currently, SYSU-30k supports both [Google drive](https://drive.google.com/drive/folders/1MTxZ4UN_mbxjByZgcAki-H10zDzzeyuJ) collection and [Baidu Pan](https://pan.baidu.com/s/1Y9phSZ5jy02szFZB_KqlyQ#list/path=%2F) (code: 1qzv) collection.


# Installation

This repo has been tested in the following environment. More precisely, this repo is a modification on the OpenSelfSup. Installation and preparation follow that repo. Please acknowledge the great work of the team of OpenSelfSup.

For object detection and instance segmentation tasks, this repo follows OpenSelfSup and uses Detectron2. Thanks for their outstanding contributions.

Pytorch1.9

[OpenSelfSup](https://github.com/open-mmlab/OpenSelfSup)


[detectron2](https://github.com/facebookresearch/detectron2)
