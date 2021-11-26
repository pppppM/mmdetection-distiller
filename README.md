# mmdetection-distiller
### The repo will be deprecated !!!!!
### If you want to distill model in OpenMMLab related repos, could join the wechat group!!
<img src="https://github.com/pppppM/mmsegmentation-distiller/blob/master/WechatIMG39.jpeg" width="150" height="200" ><br/>

This project is based on mmdetection(v-2.9.0), all the usage is the same as [mmdetection](https://mmdetection.readthedocs.io/en/latest/) including training , test and so on.

# Distiller Zoo



- [x] [Channel-wise Distillation for Semantic Segmentation](https://github.com/pppppM/mmdetection-distiller/tree/master/configs/distillers/cwd)
- [ ] [Improve Object Detection with Feature-based Knowledge Distillation: Towards Accurate and Efficient Detectors](https://openreview.net/forum?id=uKhGRvM8QNH)



# Installation

* Set up a new conda environment: `conda create -n distiller python=3.7`

* Install pytorch

* [Install mmcv ( 1.2.4 <= mmcv-full < 1.3 )](https://github.com/open-mmlab/mmcv#installation)

* Install mmdetection-distiller

  ```bash
  git clone https://github.com/pppppM/mmdetection-distiller.git
  cd mmdetection-distiller
  pip install -r requirements/build.txt
  pip install -v -e .
  ```

# Train

```
#single GPU
python tools/train.py configs/distillers/cwd/cwd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py

#multi GPU
bash tools/dist_train.sh configs/distillers/cwd/cwd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py 8
```

# Test

```
#single GPU
python tools/test.py configs/distillers/cwd/cwd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py $CHECKPOINT --eval bbox

#multi GPU
bash tools/dist_train.sh configs/distillers/cwd/cwd_retina_rx101_64x4d_distill_retina_r50_fpn_2x_coco.py $CHECKPOINT 8 --eval bbox
```

# Lisence

This project is released under the [Apache 2.0 license](LICENSE).
