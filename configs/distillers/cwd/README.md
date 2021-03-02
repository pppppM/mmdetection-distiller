# Channel-wise Distillation for Semantic Segmentation

## Introduction

[ALGORITHM]

We provide the config files for CWD: [Channel-wise Distillation for Semantic Segmentation](https://arxiv.org/abs/2011.13256).

```BibTeX
@inproceedings{shu2020cwd,
  title={Channel-wise Distillation for Semantic Segmentation},
  author={Shu, Changyong and Liu, Yifan and Gao, Jianfei and Xu, Lin and Shen, Chunhua},
}
```

## Results and Models

| | Backbone | Detector | Lr schd | box AP | Config | Download |
|:------:|:------:|:--------:|:-------:|:------:|:------:|:------:| 
| Teacher | X-101-64x4d-FPN | RetinaNet  |2x | 40.8 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_x101_64x4d_fpn_2x_coco.py) |[model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131-bca068ab.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_x101_64x4d_fpn_2x_coco/retinanet_x101_64x4d_fpn_2x_coco_20200131_114833.log.json) | 
| Student | R-50-FPN | RetinaNet  |2x | 37.4 | [config](https://github.com/open-mmlab/mmdetection/tree/master/configs/retinanet/retinanet_r50_fpn_2x_coco.py) |  [model](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131-fdb43119.pth) &#124; [log](http://download.openmmlab.com/mmdetection/v2.0/retinanet/retinanet_r50_fpn_2x_coco/retinanet_r50_fpn_2x_coco_20200131_114738.log.json) |
| CWD | R-50-FPN | RetinaNet  |2x | 40.5 | [config](https://github.com/pppppM/mmdetection-distiller/blob/master/configs/distillers/cwd/cwd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py) 
