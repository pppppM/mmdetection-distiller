# Copyright (c) 2019 Western Digital Corporation or its affiliates.

import warnings
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, normal_init
from mmcv.runner import force_fp32

from mmdet.core import (build_anchor_generator, build_assigner,
                        build_bbox_coder, build_sampler, images_to_levels,
                        multi_apply, multiclass_nms)
from ..builder import HEADS, build_loss
from .base_dense_head import BaseDenseHead
from .dense_test_mixins import BBoxTestMixin


@HEADS.register_module()
class YOLOV5Head(BaseDenseHead, BBoxTestMixin):
    """YOLOV3Head Paper link: https://arxiv.org/abs/1804.02767.

    Args:
        num_classes (int): The number of object classes (w/o background)
        in_channels (List[int]): Number of input channels per scale.
        out_channels (List[int]): The number of output channels per scale
            before the final 1x1 layer. Default: (1024, 512, 256).
        anchor_generator (dict): Config dict for anchor generator
        bbox_coder (dict): Config of bounding box coder.
        featmap_strides (List[int]): The stride of each scale.
            Should be in descending order. Default: (32, 16, 8).
        one_hot_smoother (float): Set a non-zero value to enable label-smooth
            Default: 0.
        conv_cfg (dict): Config dict for convolution layer. Default: None.
        norm_cfg (dict): Dictionary to construct and config norm layer.
            Default: dict(type='BN', requires_grad=True)
        act_cfg (dict): Config dict for activation layer.
            Default: dict(type='LeakyReLU', negative_slope=0.1).
        loss_cls (dict): Config of classification loss.
        loss_conf (dict): Config of confidence loss.
        loss_xy (dict): Config of xy coordinate loss.
        loss_wh (dict): Config of wh coordinate loss.
        train_cfg (dict): Training config of YOLOV3 head. Default: None.
        test_cfg (dict): Testing config of YOLOV3 head. Default: None.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 anchor_generator=dict(
                     type='YOLOAnchorGenerator',
                     base_sizes=[[(116, 90), (156, 198), (373, 326)],
                                 [(30, 61), (62, 45), (59, 119)],
                                 [(10, 13), (16, 30), (33, 23)]],
                     strides=[32, 16, 8]),
                 bbox_coder=dict(type='YOLOBBoxCoder'),
                 featmap_strides=[32, 16, 8],
                 conf_smooth=0.1,
                 one_hot_smoother=0.,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN', requires_grad=True),
                 act_cfg=dict(type='LeakyReLU', negative_slope=0.1),
                 loss_cls=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_conf=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox=dict(
                     type='CIoULoss',
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None):
        super(YOLOV5Head, self).__init__()
        # Check params
        assert (len(in_channels)  == len(featmap_strides))

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.featmap_strides = featmap_strides
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            if hasattr(self.train_cfg, 'sampler'):
                sampler_cfg = self.train_cfg.sampler
            else:
                sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.one_hot_smoother = one_hot_smoother
        self.conf_smooth = conf_smooth
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.anchor_generator = build_anchor_generator(anchor_generator)

        self.loss_cls = build_loss(loss_cls)
        self.loss_conf = build_loss(loss_conf)
        self.loss_bbox = build_loss(loss_bbox)
        # self.loss_wh = build_loss(loss_wh)
        # usually the numbers of anchors for each level are the same
        # except SSD detectors
        self.num_anchors = self.anchor_generator.num_base_anchors[0]
        assert len(
            self.anchor_generator.num_base_anchors) == len(featmap_strides)
        self._init_layers()

    @property
    def num_levels(self):
        return len(self.featmap_strides)

    @property
    def num_attrib(self):
        """int: number of attributes in pred_map, bboxes (4) +
        objectness (1) + num_classes"""

        return 5 + self.num_classes

    def _init_layers(self):
        
        self.convs_pred = nn.ModuleList()
        for i in range(self.num_levels):
            
            conv_pred = nn.Conv2d(self.in_channels[i],
                                  self.num_anchors * self.num_attrib, 1)
            bias = conv_pred.bias.view(self.num_anchors, -1)
            stride = conv_pred.stride[0]
            bias[:, 4] += math.log(8 / (608 / stride) ** 2)
            bias[:, 5:] += math.log(0.6 / (self.num_classes - 0.99))
            conv_pred.bias =torch.nn.Parameter(bias.view(-1), requires_grad=True)
            self.convs_pred.append(conv_pred)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.convs_pred:
            normal_init(m, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple[Tensor]: A tuple of multi-level predication map, each is a
                4D-tensor of shape (batch_size, 5+num_classes, height, width).
        """

        assert len(feats) == self.num_levels
        pred_maps = []
        for i in range(self.num_levels):
            x = feats[i]
            pred_map = self.convs_pred[i](x)
            pred_maps.append(pred_map)

        return tuple(pred_maps),

    @force_fp32(apply_to=('pred_maps', ))
    def get_bboxes(self,
                   pred_maps,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            pred_maps (list[Tensor]): Raw predictions for a batch of images.
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        result_list = []
        num_levels = len(pred_maps)
        for img_id in range(len(img_metas)):
            pred_maps_list = [
                pred_maps[i][img_id].detach() for i in range(num_levels)
            ]
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self._get_bboxes_single(pred_maps_list, scale_factor,
                                                cfg, rescale, with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           pred_maps_list,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            pred_maps_list (list[Tensor]): Prediction maps for different scales
                of each single image in the batch.
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(pred_maps_list) == self.num_levels
        multi_lvl_bboxes = []
        multi_lvl_cls_scores = []
        multi_lvl_conf_scores = []
        num_levels = len(pred_maps_list)
        featmap_sizes = [
            pred_maps_list[i].shape[-2:] for i in range(num_levels)
        ]
        multi_lvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, pred_maps_list[0][0].device)
        for i in range(self.num_levels):
            # get some key info for current scale
            pred_map = pred_maps_list[i]
            stride = self.featmap_strides[i]

            # (h, w, num_anchors*num_attrib) -> (h*w*num_anchors, num_attrib)
            pred_map = pred_map.permute(1, 2, 0).reshape(-1, self.num_attrib)

            # pred_map[..., :2] = torch.sigmoid(pred_map[..., :2])
            bbox_pred = self.bbox_coder.decode(multi_lvl_anchors[i],
                                               pred_map[..., :4], stride)
            # conf and cls
            conf_pred = torch.sigmoid(pred_map[..., 4]).view(-1)
            cls_pred = torch.sigmoid(pred_map[..., 5:]).view(
                -1, self.num_classes)  # Cls pred one-hot.

            # Filtering out all predictions with conf < conf_thr
            conf_thr = cfg.get('conf_thr', -1)
            conf_inds = conf_pred.ge(conf_thr).nonzero().flatten()
            bbox_pred = bbox_pred[conf_inds, :]
            cls_pred = cls_pred[conf_inds, :]
            conf_pred = conf_pred[conf_inds]

            # Get top-k prediction
            nms_pre = cfg.get('nms_pre', -1)
            if 0 < nms_pre < conf_pred.size(0):
                _, topk_inds = conf_pred.topk(nms_pre)
                bbox_pred = bbox_pred[topk_inds, :]
                cls_pred = cls_pred[topk_inds, :]
                conf_pred = conf_pred[topk_inds]

            # Save the result of current scale
            multi_lvl_bboxes.append(bbox_pred)
            multi_lvl_cls_scores.append(cls_pred)
            multi_lvl_conf_scores.append(conf_pred)

        # Merge the results of different scales together
        multi_lvl_bboxes = torch.cat(multi_lvl_bboxes)
        multi_lvl_cls_scores = torch.cat(multi_lvl_cls_scores)
        multi_lvl_conf_scores = torch.cat(multi_lvl_conf_scores)

        if with_nms and (multi_lvl_conf_scores.size(0) == 0):
            return torch.zeros((0, 5)), torch.zeros((0, ))

        if rescale:
            multi_lvl_bboxes /= multi_lvl_bboxes.new_tensor(scale_factor)

        # In mmdet 2.x, the class_id for background is num_classes.
        # i.e., the last column.
        padding = multi_lvl_cls_scores.new_zeros(multi_lvl_cls_scores.shape[0],
                                                 1)
        multi_lvl_cls_scores = torch.cat([multi_lvl_cls_scores, padding],
                                         dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                multi_lvl_bboxes,
                multi_lvl_cls_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=multi_lvl_conf_scores)
            return det_bboxes, det_labels
        else:
            return (multi_lvl_bboxes, multi_lvl_cls_scores,
                    multi_lvl_conf_scores)

    @force_fp32(apply_to=('pred_maps', ))
    def loss(self,
             pred_maps,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            pred_maps (list[Tensor]): Prediction map for each scale level,
                shape (N, num_anchors * num_attrib, H, W)
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        num_imgs = len(img_metas)
        device = pred_maps[0][0].device
        featmap_sizes = [
            pred_maps[i].shape[-2:] for i in range(self.num_levels)
        ]
        multi_level_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device)
        anchor_list = [anchors_lvl.repeat(num_imgs,1) for anchors_lvl in multi_level_anchors]

        target_maps_list, pos_inds_list = self.get_targets(
           featmap_sizes, gt_bboxes, gt_labels)

        losses_cls, losses_conf, losses_bbox = multi_apply(
            self.loss_single, pred_maps, anchor_list,
            target_maps_list, pos_inds_list,self.featmap_strides,
            [gt_bboxes for _ in target_maps_list])

        return dict(
            loss_cls=losses_cls ,
            loss_conf=losses_conf,
            loss_bbox=losses_bbox)

    def loss_single(self, pred_map,anchors_lvl, target_map, pos_inds,stride,gt):
        """Compute loss of a single image from a batch.

        Args:
            pred_map (Tensor): Raw predictions for a single level.
            target_map (Tensor): The Ground-Truth target for a single level.
            neg_map (Tensor): The negative masks for a single level.

        Returns:
            tuple:
                loss_cls (Tensor): Classification loss.
                loss_conf (Tensor): Confidence loss.
                loss_xy (Tensor): Regression loss of x, y coordinate.
                loss_wh (Tensor): Regression loss of w, h coordinate.
        """

        num_imgs = len(pred_map)
        # import pdb; pdb.set_trace()
        pred_map = pred_map.permute(0, 2, 3,
                                    1).reshape( -1, self.num_attrib)
        
        pred_bboxes = self.bbox_coder.decode(anchors_lvl[pos_inds],pred_map[pos_inds,:4],stride)

        pred_conf = pred_map[..., 4]
        pred_label = pred_map[pos_inds, 5:]


        target_bboxes = self.bbox_coder.decode_target(anchors_lvl[pos_inds],target_map[:,2:6],stride)

        target_label = target_map[..., 6].long()
        loss_bbox = self.loss_bbox(pred_bboxes,target_bboxes,reduction_override='none')
        target_pos_conf = (1 - self.conf_smooth) + self.conf_smooth * (1 - loss_bbox.detach()).clamp(0)
        
        
        neg_inds = torch.ones_like(pred_conf).scatter_(0,pos_inds,0).bool()
        target_neg_conf = torch.zeros_like(pred_conf[neg_inds])
        pred_conf = torch.cat((pred_conf[pos_inds],pred_conf[neg_inds]),dim=0)
        target_conf = torch.cat((target_pos_conf,target_neg_conf),dim=0)

        
        loss_bbox = loss_bbox.mean() * num_imgs
        loss_cls = self.loss_cls(pred_label, target_label) * num_imgs
        loss_conf = self.loss_conf(
            pred_conf, target_conf) * num_imgs


        return loss_cls, loss_conf, loss_bbox

    def get_targets(self, featmap_sizes, gt_bboxes_list,
                    gt_labels_list):
        """Compute target maps for anchors in multiple images.

        Args:
            anchor_list (list[list[Tensor]]): Multi level anchors of each
                image. The outer list indicates images, and the inner list
                corresponds to feature levels of the image. Each element of
                the inner list is a tensor of shape (num_total_anchors, 4).
            responsible_flag_list (list[list[Tensor]]): Multi level responsible
                flags of each image. Each element is a tensor of shape
                (num_total_anchors, )
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image.
            gt_labels_list (list[Tensor]): Ground truth labels of each box.

        Returns:
            tuple: Usually returns a tuple containing learning targets.
                - target_map_list (list[Tensor]): Target map of each level.
                - neg_map_list (list[Tensor]): Negative map of each level.
        """
        num_imgs = len(gt_bboxes_list)
        

        # anchor number of multi levels
        # num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        featmap_sizes_list = [featmap_sizes for _ in range(num_imgs)]
        results = multi_apply(self._get_targets_single,featmap_sizes_list,
                              gt_bboxes_list,
                              gt_labels_list)
        
        all_target, _ = results
        target_lvl_list = []
        target_inds_lvl_list = []
        for lvl in range(len(featmap_sizes)):
            target_lvl = []
            target_inds_lvl = []
            feat_h, feat_w = featmap_sizes[lvl]
            for i in range(num_imgs):
                per_img_target = all_target[i]
                per_img_target_lvl = per_img_target[lvl]
                grid_i,grid_j = per_img_target_lvl[:,0],per_img_target_lvl[:,1]
                anchor_index = per_img_target_lvl[:,-1]
                target_lvl.append(per_img_target_lvl)
                # import pdb; pdb.set_trace()
                target_inds_lvl.append(i * feat_h * feat_w * 3 \
                                        + grid_j * feat_w *3  \
                                        + grid_i * 3 + anchor_index)
            # import pdb; pdb.set_trace()
            target_lvl = torch.cat(target_lvl)
            target_inds_lvl = torch.cat(target_inds_lvl)
            
            target_lvl_list.append(target_lvl)
            target_inds_lvl_list.append(target_inds_lvl.long())


        return target_lvl_list, target_inds_lvl_list

    def _get_targets_single(self, featmap_size,gt_bboxes,
                            gt_labels):
        """Generate matching bounding box prior and converted GT.

        Args:
            anchors (list[Tensor]): Multi-level anchors of the image.
            responsible_flags (list[Tensor]): Multi-level responsible flags of
                anchors
            gt_bboxes (Tensor): Ground truth bboxes of single image.
            gt_labels (Tensor): Ground truth labels of single image.

        Returns:
            tuple:
                target_map (Tensor): Predication target map of each
                    scale level, shape (num_total_anchors,
                    5+num_classes)
                neg_map (Tensor): Negative map of each scale level,
                    shape (num_total_anchors,)
        """

        
        base_sizes = gt_bboxes.new_tensor(self.anchor_generator.base_sizes)
        gts = torch.cat([gt_bboxes,gt_labels.unsqueeze(-1)],dim=1)
        num_anchors = base_sizes.size(1)
        num_gts = gts.size(0)
        anchor_index = torch.arange(num_anchors,device=gts.device).float().view(num_anchors,1).repeat(1,num_gts)
        anchor_gt_pair = torch.cat([gts.repeat(num_anchors,1,1) , anchor_index.unsqueeze(-1)],dim=2)
        

        off = gt_bboxes.new_tensor([[0,0],[1,0],[0,1],[-1,0],[0,-1]])

        target_lvl_list = []

        for size ,fsize, stride in zip(base_sizes,featmap_size,self.featmap_strides):
            feat_h, feat_w = fsize
            # import pdb; pdb.set_trace()
            scale_gt_bboxes = gt_bboxes / stride
            scale_base_sizes = size / stride
            if len(gt_bboxes):
                scale_gt_bboxes_wh = scale_gt_bboxes[:,-2:] - scale_gt_bboxes[:,:2]
                gt_anchor_ratio = (scale_gt_bboxes_wh.unsqueeze(0)/scale_base_sizes.unsqueeze(1))
                pos_inds = torch.max(gt_anchor_ratio , 1. / gt_anchor_ratio).max(2) [0] < 4.0

                pos_grids = anchor_gt_pair[pos_inds]

                pos_grids[:,:4] = pos_grids[:,:4]/stride

                pos_grids_cx = ((pos_grids[:, 0] + pos_grids[:, 2]) * 0.5)
                pos_grids_cy = ((pos_grids[:, 1] + pos_grids[:, 3]) * 0.5)

                right = (pos_grids_cx % 1.0 < 0.5) & (pos_grids_cx > 1)
                bottom = (pos_grids_cy % 1.0 < 0.5) & (pos_grids_cy > 1)
                left =  ((feat_h - pos_grids_cx) % 1.0 < 0.5) & (feat_h - pos_grids_cx > 1)
                top =  ((feat_w - pos_grids_cy) % 1.0 < 0.5) & (feat_w - pos_grids_cy > 1)
                
                nearest_inds = torch.stack((torch.ones_like(right),right,bottom,left,top))
                nearest_grids = pos_grids.repeat(5,1,1)[nearest_inds]
                nearest_offsets =  off.unsqueeze(1).repeat(1,len(pos_grids),1)[nearest_inds]
                nearest_offsets_x = nearest_offsets[:,0]
                nearest_offsets_y = nearest_offsets[:,1]
            else:

                nearest_grids = anchor_gt_pair[0]
                nearest_offsets = 0
                nearest_offsets_x = 0
                nearest_offsets_y = 0
            
            
            nearest_grids_cx = ((nearest_grids[:, 0] + nearest_grids[:, 2]) * 0.5)
            nearest_grids_cy = ((nearest_grids[:, 1] + nearest_grids[:, 3]) * 0.5)
            try:
                grid_cx = (nearest_grids_cx + nearest_offsets_x)
                grid_cy = (nearest_grids_cy + nearest_offsets_y)
            except:
                import pdb; pdb.set_trace()
            # import pdb;pdb.set_trace()
            grid_i = grid_cx.long().clamp(0 , feat_w-1)
            grid_j = grid_cy.long().clamp(0 , feat_h-1)
            grid_offset_x = nearest_grids_cx - grid_i
            grid_offset_y = nearest_grids_cy - grid_j
            grid_w,grid_h = (nearest_grids[:,2:4] - nearest_grids[:,:2]).T
            grid_cls = nearest_grids[:,4]
            grid_anchor_index = nearest_grids[:,5]

            target_lvl = torch.stack((grid_i,grid_j,
                                      grid_offset_x,grid_offset_y,
                                      grid_w,grid_h,
                                      grid_cls,grid_anchor_index),dim=1)
            # import pdb; pdb.set_trace()
            target_lvl_list.append(target_lvl)

        return target_lvl_list,target_lvl_list


    def aug_test(self, feats, img_metas, rescale=False):
        """Test function with test time augmentation.

        Args:
            feats (list[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains features for all images in the batch.
            img_metas (list[list[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch. each dict has image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[ndarray]: bbox results of each class
        """
        return self.aug_test_bboxes(feats, img_metas, rescale=rescale)
