import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import Scale, normal_init,ConvModule,build_norm_layer
from mmdet.core import distance2bbox, force_fp32, multi_apply, multiclass_nms,bbox_overlaps
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
import numpy as np
INF = 1e8


@HEADS.register_module()
class AutoAssignHead(AnchorFreeHead):
    """Anchor-free head used in `FCOS <https://arxiv.org/abs/1904.01355>`_.

    The FCOS head does not use anchor boxes. Instead bounding boxes are
    predicted at each pixel and a centerness measure is used to supress
    low-quality predictions.
    Here norm_on_bbox, centerness_on_reg, dcn_on_last_conv are training
    tricks used in official repo, which will bring remarkable mAP gains
    of up to 4.9. Please see https://github.com/tianzhi0549/FCOS for
    more detail.

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        strides (list[int] | list[tuple[int, int]]): Strides of points
            in multiple feature levels. Default: (4, 8, 16, 32, 64).
        regress_ranges (tuple[tuple[int, int]]): Regress range of multiple
            level points.
        center_sampling (bool): If true, use center sampling. Default: False.
        center_sample_radius (float): Radius of center sampling. Default: 1.5.
        norm_on_bbox (bool): If true, normalize the regression targets
            with FPN strides. Default: False.
        centerness_on_reg (bool): If true, position centerness on the
            regress branch. Please refer to https://github.com/tianzhi0549/FCOS/issues/89#issuecomment-516877042.
            Default: False.
        conv_bias (bool | str): If specified as `auto`, it will be decided by the
            norm_cfg. Bias of conv will be set as True if `norm_cfg` is None, otherwise
            False. Default: "auto".
        loss_cls (dict): Config of classification loss.
        loss_bbox (dict): Config of localization loss.
        loss_centerness (dict): Config of centerness loss.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: norm_cfg=dict(type='GN', num_groups=32, requires_grad=True).

    Example:
        >>> self = FCOSHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_score, bbox_pred, centerness = self.forward(feats)
        >>> assert len(cls_score) == len(self.scales)
    """  # noqa: E501

    def __init__(self,
                 num_classes,
                 in_channels,
                 regress_ranges=((-1, 64), (64, 128), (128, 256), (256, 512),
                                 (512, INF)),
                 center_sampling=False,
                 center_sample_radius=1.5,
                 norm_on_bbox=False,
                 centerness_on_reg=False,
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='IoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 **kwargs):
        self.regress_ranges = regress_ranges
        self.center_sampling = center_sampling
        self.center_sample_radius = center_sample_radius
        self.norm_on_bbox = norm_on_bbox
        self.centerness_on_reg = centerness_on_reg
        self.background_label = num_classes
        super().__init__(
            num_classes,
            in_channels,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            norm_cfg=norm_cfg,
            **kwargs)
        self.loss_centerness = build_loss(loss_centerness)
        

    def _init_gumbel_convs(self):
        """Initialize classification conv layers of the head."""
        self.gumbel_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and i == self.stacked_convs - 1:
                conv_cfg = dict(type='DCNv2')
            else:
                conv_cfg = self.conv_cfg
            self.gumbel_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    1,
                    stride=1,
                    padding=0,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                    bias=self.conv_bias))

    def _init_layers(self):
        """Initialize layers of the head."""
        super()._init_layers()
        self._init_gumbel_convs()
        self.conv_centerness = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        self.conv_gumbel = nn.Conv2d(self.feat_channels, 1, 1, padding=0)
        self.scales = nn.ModuleList([Scale(1.0) for _ in self.strides])
        _,self.gumbel_norm = build_norm_layer(self.norm_cfg,self.feat_channels)
    def init_weights(self):
        """Initialize weights of the head."""
        super().init_weights()
        normal_init(self.conv_centerness, std=0.01)

    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple:
                cls_scores (list[Tensor]): Box scores for each scale level, \
                    each is a 4D-tensor, the channel number is \
                    num_points * num_classes.
                bbox_preds (list[Tensor]): Box energies / deltas for each \
                    scale level, each is a 4D-tensor, the channel number is \
                    num_points * 4.
                centernesses (list[Tensor]): Centerss for each scale level, \
                    each is a 4D-tensor, the channel number is num_points * 1.
        """
        return multi_apply(self.forward_single, feats, self.scales,
                           self.strides)

    def forward_single(self, x, scale, stride):
        """Forward features of a single scale levle.

        Args:
            x (Tensor): FPN feature maps of the specified stride.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.
            stride (int): The corresponding stride for feature maps, only
                used to normalize the bbox prediction when self.norm_on_bbox
                is True.

        Returns:
            tuple: scores for each class, bbox predictions and centerness \
                predictions of input feature maps.
        """
        # import pdb;pdb.set_trace()
        # cls_score, bbox_pred, cls_feat, reg_feat = super().forward_single(x)
        cls_feat = x
        reg_feat = x

        for cls_layer in self.cls_convs:
            cls_feat = cls_layer(cls_feat)
        cls_score = self.conv_cls(cls_feat)

        for reg_layer in self.reg_convs:
            reg_feat = reg_layer(reg_feat)
        bbox_pred = self.conv_reg(reg_feat)
        # import pdb;pdb.set_trace()
        if self.centerness_on_reg:
            centerness = self.conv_centerness(reg_feat)
        else:
            centerness = self.conv_centerness(cls_feat)
        # scale the bbox_pred of different level
        # float to avoid overflow when enabling FP16
        bbox_pred = scale(bbox_pred).float()
        if self.norm_on_bbox:
            bbox_pred = F.relu(bbox_pred)
            if not self.training:
                bbox_pred *= stride
        else:
            bbox_pred = bbox_pred.exp()
        return x.detach() , cls_score, bbox_pred,centerness

    @force_fp32(apply_to=('feats','cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             feats,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        """Compute loss of the head.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level,
                each is a 4D-tensor, the channel number is
                num_points * num_classes.
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level, each is a 4D-tensor, the channel number is
                num_points * 4.
            centernesses (list[Tensor]): Centerss for each scale level, each
                is a 4D-tensor, the channel number is num_points * 1.
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
        assert len(cls_scores) == len(bbox_preds) == len(feats)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        featmap_sizes_tensor = cls_scores[0].new_tensor(featmap_sizes)
        all_level_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                           bbox_preds[0].device)
        num_imgs = cls_scores[0].size(0)

        num_points = [center.size(0)  for center in all_level_points]
        # import pdb;pdb.set_trace()
        points_img_index = [x for x in num_points for y in range(num_imgs)  ]
        per_fpn_begin_index = [0] + num_points[:-1]
        num_points_tensor = cls_scores[0].new_tensor(torch.tensor(num_points)).long()
        per_fpn_begin_index = torch.cumsum(torch.tensor(per_fpn_begin_index),dim=0)
        per_fpn_begin_index = cls_scores[0].new_tensor(per_fpn_begin_index).long() * num_imgs
        points_img_index = torch.cat([torch.ones(size) * (i % num_imgs) for i,size in enumerate(points_img_index)])
        points_img_index = cls_scores[0].new_tensor(points_img_index).long()
        
        labels, bbox_targets = self.get_targets(all_level_points, gt_bboxes,
                                                gt_labels)
        
        
        flatten_feats = [
            feat.permute(0, 2, 3, 1).reshape(-1, feat.size(1))
            for feat in feats
        ]
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]

        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]


        flatten_feats = torch.cat(flatten_feats)
        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_centerness = torch.cat(flatten_centerness)
        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        # repeat points to align with bbox_preds
        
        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        flatten_bboxes = distance2bbox(flatten_points,flatten_bbox_preds)
        
        

        per_gt_labels = flatten_labels.split(1,dim=1)
        per_gt_bbox_targets = flatten_bbox_targets.split(1,dim=1)

        pos_loss = []
        in_neg_loss = []
        num_pos_points = ((flatten_labels >= 0) & (flatten_labels<self.background_label)).sum()
        val_point_inds = (flatten_labels >= 0) & (flatten_labels<self.background_label)
        neg_point_inds = (val_point_inds.sum(1) == 0).view(-1)
        pos_point_inds = (val_point_inds.sum(1) >0).view(-1)
        per_batch_gt_nums = sum([bbox.size(0) for bbox in gt_bboxes])
        
        assert num_pos_points > 0


        pos_point_mask = (flatten_labels >= 0) & (flatten_labels<self.background_label)
        neg_point_inds = (pos_point_mask.sum(1) == 0).view(-1)
        for gt_id,(label,bbox)  in enumerate( zip(per_gt_labels,per_gt_bbox_targets)):
            pos_inds = ((label.squeeze() >= 0)
                    & (label.squeeze() < self.background_label)).nonzero().reshape(-1)


            pos_points_img_inds = [pos_inds[points_img_index[pos_inds]==i] for i in range(num_imgs)]

            per_img_pos_losses = [self._get_single_pos_loss(flatten_feats[inds],
                                                    flatten_cls_scores[inds],
                                                    flatten_bbox_preds[inds],
                                                    flatten_centerness[inds],
                                                    flatten_points[inds],
                                                    label[inds],
                                                    bbox[inds]) for inds in pos_points_img_inds]
            pos_loss.append(sum(per_img_pos_losses))

            


        for i,gt_bbox in enumerate(gt_bboxes):
            # per_img_points = flatten_points[points_img_index==i]
            per_img_points = pos_point_inds[points_img_index==i]
            # label = flatten_labels[(points_img_index==i) & (pos_point_inds>0)]
            # import pdb;pdb.set_trace()
            pos_inds = (per_img_points).nonzero().reshape(-1)


            per_img_cls = flatten_cls_scores[points_img_index==i]
            per_img_centerness = flatten_centerness[points_img_index==i]
            per_img_bboxes = flatten_bboxes[points_img_index==i]
            object_box_iou = bbox_overlaps(gt_bbox,per_img_bboxes)
            object_box_iou = object_box_iou[:,pos_inds]
            # t1 = 0.6
            # iou = object_box_iou.max(
            #             dim=1, keepdim=True).values
            # import pdb;pdb.s
            # et_trace()
            fiou = 1/(1 - object_box_iou.max(0)[0])
            fiou_min = fiou.min()
            fiou_max = fiou.max()
            fiou_nrom = (fiou - fiou_min) / (fiou_max - fiou_min).clamp(min=1e-12)
            

            # object_box_prob = ((object_box_iou - t1) /
            #                     (t2 - t1)).clamp(
            #                         min=0, max=1)

            # box_prob = object_box_prob.max(0).values.detach()
            neg_prob = per_img_cls[pos_inds].sigmoid() * (1-fiou_nrom.detach()).unsqueeze(-1) *per_img_centerness[pos_inds].sigmoid().unsqueeze(-1)
            per_img_neg_loss = 0.75 * neg_prob ** 2 * F.binary_cross_entropy(neg_prob,torch.zeros_like(neg_prob),reduction='none')
            in_neg_loss.append(per_img_neg_loss.sum())
            
            # import pdb;pdb.set_trace()
        
        loss_in_neg = sum(in_neg_loss) / pos_point_mask.sum()
        neg_prob = flatten_cls_scores[neg_point_inds].sigmoid()
        loss_out_neg =  0.75 * (neg_prob ** 2) * F.binary_cross_entropy(neg_prob,torch.zeros_like(neg_prob),reduction='none')
        loss_out_neg = loss_out_neg.sum() / pos_point_mask.sum() 
        
        # import pdb;pdb.set_trace()
        loss_pos = sum(pos_loss)/per_batch_gt_nums  #/per_batch_gt_nums#.sum() / (per_points_pos_cls_loss.shape[1] * num_imgs)
       
        return dict(
            loss_in_neg=loss_in_neg,
            loss_out_neg=loss_out_neg,
            loss_pos=loss_pos,
            )


    @force_fp32(apply_to=('flatten_feats','cls_scores','centernesses', 'flatten_bbox_preds'))
    def _get_single_pos_loss(self,
                           flatten_feats,
                           cls_scores,
                           flatten_bbox_preds,
                           centernesses,
                           flatten_points,
                           flatten_labels,
                           flatten_bbox_targets,
                           ):
        # import pdb;pdb.set_tra?ce()
        flatten_labels = flatten_labels.view(-1)
        label_inds = (flatten_labels >= 0).nonzero().reshape(-1)
        # import pdb;pdb.set_trace()
        flatten_bbox_targets = flatten_bbox_targets.view(-1,4)
        bg_class_ind = self.num_classes
        
        
        unique_label = torch.unique(flatten_labels).squeeze()
        
        pos_inds = ((flatten_labels >= 0)
                    & (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = len(pos_inds)
        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        
            
        if num_pos > 0 :
            

            pos_points = flatten_points[pos_inds]
            

            
            
            pos_decoded_bbox_preds = distance2bbox(pos_points, pos_bbox_preds)
            pos_bbox_targets = flatten_bbox_targets[pos_inds]
            pos_decoded_target_preds = distance2bbox(pos_points,pos_bbox_targets)
            # import pdb;pdb.set_trace()
            loss_bbox = self.loss_bbox(
                    pos_decoded_bbox_preds,
                    pos_decoded_target_preds,
                    reduction_override='none') 
            # import pdb;pdb.set_trace()
            cls_prob = cls_scores[:,unique_label].sigmoid()
            bbox_prob = torch.exp(-loss_bbox * 5.0)
            pos_prob = cls_prob * bbox_prob * centernesses.sigmoid()
            # gumbel_weight = pos_prob.detach().view(1,-1)
            # K = 5000
            # gumbel_weight = gumbel_weight.repeat(K,1)
            # gumbel_weight = F.gumbel_softmax(gumbel_weight,hard=True).sum(0)
            # prob_sum = (gumbel_weight*pos_prob).sum() / K

            # pos_loss = 0.25 * F.binary_cross_entropy(prob_sum,torch.ones_like(prob_sum),reduction='sum')

            w_plus = torch.softmax(3*pos_prob,dim=-1).detach()
            prob_sum = (w_plus*pos_prob).sum()
            pos_loss = 0.25 * (1-prob_sum)**2 * F.binary_cross_entropy(prob_sum,torch.ones_like(prob_sum),reduction='sum')
            # pos_loss = 0.25 *  F.binary_cross_entropy(prob_sum,torch.ones_like(prob_sum),reduction='sum')
            
            
        else:
            pos_loss = cls_scores[pos_inds].sum()
            # loss_bbox = pos_bbox_preds.sum()
            # loss_centerness = centernesses[pos_inds].sum()

        return pos_loss
    @force_fp32(apply_to=('cls_scores', 'bbox_preds','centernesses'))
    def get_bboxes(self,
                   feats,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=None):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                Has shape (N, num_points * num_classes, H, W)
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W)
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W)
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used
            rescale (bool): If True, return boxes in original image space

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple. \
                The first item is an (n, 5) tensor, where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1. The second item is a \
                (n,) tensor where each item is the predicted class label of \
                the corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)
        # import pdb;pdb.set_trace()
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]

            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(cls_score_list,
                                                 bbox_pred_list,
                                                 centerness_pred_list,
                                                 mlvl_points, img_shape,
                                                 scale_factor, cfg, rescale)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                Has shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.

        Returns:
            Tensor: Labeled boxes in shape (n, 5), where the first 4 columns \
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the \
                5-th column is a score between 0 and 1.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred,  centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:,None]  ).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_centerness = torch.cat(mlvl_centerness)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        # import pdb;pdb.set_trace()
        det_bboxes, det_labels = multiclass_nms(
            mlvl_bboxes,
            mlvl_scores,
            cfg.score_thr,
            cfg.nms,
            cfg.max_per_img,
            score_factors=mlvl_centerness)
        # import pdb;pdb.set_trace()
        return det_bboxes, det_labels

    def _get_points_single(self,
                           featmap_size,
                           stride,
                           dtype,
                           device,
                           flatten=False):
        """Get points according to feature map sizes."""
        y, x = super()._get_points_single(featmap_size, stride, dtype, device)
        z = torch.log2(y.reshape(-1).new_tensor(stride)[None].expand_as(y.reshape(-1))) - 3 - stride //2 
        # import pdb;pdb.set_trace()
        points = torch.stack((x.reshape(-1) * stride, y.reshape(-1) * stride,z),
                             dim=-1) + stride // 2
        return points

    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        """Compute regression, classification and centerss targets for points
        in multiple images.

        Args:
            points (list[Tensor]): Points of each fpn level, each has shape
                (num_points, 2).
            gt_bboxes_list (list[Tensor]): Ground truth bboxes of each image,
                each has shape (num_gt, 4).
            gt_labels_list (list[Tensor]): Ground truth labels of each box,
                each has shape (num_gt,).

        Returns:
            tuple:
                concat_lvl_labels (list[Tensor]): Labels of each level. \
                concat_lvl_bbox_targets (list[Tensor]): BBox targets of each \
                    level.
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)
        
       
        concat_points = torch.cat(points, dim=0)

        # the number of points per img, per lvl
        num_points = [center.size(0) for center in points]

        # get labels and bbox_targets of each image
        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=concat_points,
            regress_ranges=None,
            num_points_per_lvl=num_points)

        # split to per img, per level
        labels_list = [labels.split(num_points, 0) for labels in labels_list]

        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        per_gt_pad_list = [labels[0].shape[1] for labels in labels_list]
        per_gt_pad_list = [(0,0,0,max(per_gt_pad_list) - pad,0,0) for pad in per_gt_pad_list]
        
        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([F.pad(labels[i],pad,'constant',-1) for labels,pad in zip(labels_list,per_gt_pad_list)]))
            bbox_targets = torch.cat(
                [F.pad(bbox_targets[i],pad,'constant',-1) for bbox_targets,pad in zip(bbox_targets_list,per_gt_pad_list)])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)

        
        
        return concat_lvl_labels,concat_lvl_bbox_targets  


    def _get_target_single(self, gt_bboxes, gt_labels, points, regress_ranges,
                           num_points_per_lvl):
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.background_label), \
                   gt_bboxes.new_zeros((num_points, 4))
        
        areas = (gt_bboxes[:, 2] - gt_bboxes[:, 0]) * (
            gt_bboxes[:, 3] - gt_bboxes[:, 1])

        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 4)
        xs, ys = points[:, 0], points[:, 1]
        xs = xs[:, None].expand(num_points, num_gts)
        ys = ys[:, None].expand(num_points, num_gts)

        left = xs - gt_bboxes[..., 0]
        right = gt_bboxes[..., 2] - xs
        top = ys - gt_bboxes[..., 1]
        bottom = gt_bboxes[..., 3] - ys
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        
        


        label_targets = gt_labels.view(-1,1)[None].repeat(num_points,1,1)
        label_targets[inside_gt_bbox_mask==0,:] = self.background_label 
        

        return label_targets, bbox_targets

    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.
        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)
        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
