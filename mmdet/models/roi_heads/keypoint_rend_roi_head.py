# Modified from https://github.com/facebookresearch/detectron2/tree/master/projects/PointRend  # noqa

import torch
import torch.nn.functional as F

from mmdet.core import bbox2roi, bbox_mapping, merge_aug_masks
from mmdet.ops import point_sample, rel_roi_point_to_rel_img_point
from .. import builder
from ..builder import HEADS
from .standard_roi_head import StandardRoIHead


@HEADS.register_module()
class KeyPointRendRoIHead(StandardRoIHead):
    """`PointRend <https://arxiv.org/abs/1912.08193>`_."""

    def __init__(self, point_head, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.init_point_head(point_head)
        self.init_assigner_sampler()

    def init_assigner_sampler(self):
        """Initialize assigner and sampler."""
        pass

    def init_point_head(self, point_head):
        """Initialize ``point_head``"""
        self.point_head = builder.build_head(point_head)

    def init_weights(self, pretrained):
        """Initialize the weights in head.

        Args:
            pretrained (str, optional): Path to pre-trained weights.
        """
        super().init_weights(pretrained)
        self.point_head.init_weights()

    def forward_train(self,
                      x,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks_roi=None,
                      gt_masks_bitmap=None,
                      ):
        """
        Args:
            x (list[Tensor]): list of multi-level img features.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.
            proposals (list[Tensors]): list of region proposals.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.
            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        losses = dict()
        # mask head forward and loss
        if self.with_mask:
            mask_results = self._mask_forward_train(x[0], gt_bboxes[0],
                                                    gt_masks_roi[0], gt_masks_bitmap, gt_labels[0], img_metas)
            # TODO: Support empty tensor input. #2280
            if mask_results['loss_mask'] is not None:
                losses.update(mask_results['loss_mask'])

        return losses

    def _mask_forward_train(self, x, gt_bboxes, gt_masks_roi, gt_masks_bitmap, gt_labels, img_metas):
        """Run forward function and calculate loss for mask head and point head
        in training."""
        mask_results = self.standard_mask_forward_train(x, gt_bboxes,
                                                        gt_masks_roi, gt_labels)
        if mask_results['loss_mask'] is not None:
            loss_point = self._mask_point_forward_train(
                x, gt_bboxes, mask_results['mask_pred'], gt_masks_bitmap, gt_labels,
                img_metas)
            mask_results['loss_mask'].update(loss_point)

        return mask_results

    def standard_mask_forward_train(self, x, gt_bboxes, gt_masks, gt_labels):
        """Run forward function and calculate loss for mask head in
        training."""
        if not self.share_roi_extractor:
            if gt_bboxes.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(x, rois=gt_bboxes)
        else:
            pos_inds = []
            device = bbox_feats.device
            for res in sampling_results:
                pos_inds.append(
                    torch.ones(
                        res.pos_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
                pos_inds.append(
                    torch.zeros(
                        res.neg_bboxes.shape[0],
                        device=device,
                        dtype=torch.uint8))
            pos_inds = torch.cat(pos_inds)
            if pos_inds.shape[0] == 0:
                return dict(loss_mask=None)
            mask_results = self._mask_forward(
                x, pos_inds=pos_inds, bbox_feats=bbox_feats)

        # mask_targets -> (Num, 7, 7)
        # mask_results['mask_pred'] -> (Num, Cls, 7, 7)
        loss_mask = self.mask_head.loss(mask_results['mask_pred'],
                                        gt_masks, gt_labels)

        mask_results.update(loss_mask=loss_mask, mask_targets=gt_masks)
        return mask_results

    def _mask_forward(self, x, rois=None, pos_inds=None, bbox_feats=None):
        assert ((rois is not None) ^
                (pos_inds is not None and bbox_feats is not None))
        if rois is not None:
            mask_feats = self.mask_roi_extractor([x], rois)
            # mask_feats = self.mask_roi_extractor(x[:self.mask_roi_extractor.num_inputs], rois)
            if self.with_shared_head:
                mask_feats = self.shared_head(mask_feats)
        else:
            assert bbox_feats is not None
            mask_feats = bbox_feats[pos_inds]

        mask_pred = self.mask_head(mask_feats)
        mask_results = dict(mask_pred=mask_pred, mask_feats=mask_feats)
        return mask_results

    def _mask_point_forward_train(self, x, gt_bboxes, mask_pred,
                                  gt_masks, gt_labels, img_metas):
        """Run forward function and calculate loss for point head in
        training."""
        # gt_labels = ([Num])
        # mask_pred = ([Num, cls, 7, 7])
        rel_roi_points = self.point_head.get_roi_rel_points_train(
            mask_pred, gt_labels, cfg=self.train_cfg)

        ## WARNING: if use FPN for multi layer change [x] to normal x
        if isinstance(x, torch.Tensor): x = [x]

        fine_grained_point_feats = self._get_fine_grained_point_feats(
            x, gt_bboxes, rel_roi_points, img_metas)
        coarse_point_feats = point_sample(mask_pred, rel_roi_points)
        mask_point_pred = self.point_head(fine_grained_point_feats,
                                          coarse_point_feats)
        mask_point_target = self.point_head.get_targets(
            gt_bboxes, rel_roi_points, gt_masks, img_metas, self.train_cfg)
        loss_mask_point = self.point_head.loss(mask_point_pred,
                                               mask_point_target, gt_labels)

        return loss_mask_point

    def _get_fine_grained_point_feats(self, x, rois, rel_roi_points,
                                      img_metas):
        """Sample fine grained feats from each level feature map and
        concatenate them together."""
        num_imgs = len(img_metas)
        fine_grained_feats = []
        for idx in range(self.mask_roi_extractor.num_inputs):
            feats = x[idx]
            spatial_scale = 1. / float(
                self.mask_roi_extractor.featmap_strides[idx])
            point_feats = []
            for batch_ind in range(num_imgs):
                # unravel batch dim
                feat = feats[batch_ind].unsqueeze(0)
                inds = (rois[:, 0].long() == batch_ind)
                if inds.any():
                    rel_img_points = rel_roi_point_to_rel_img_point(
                        rois[inds], rel_roi_points[inds], feat.shape[2:],
                        spatial_scale).unsqueeze(0)
                    point_feat = point_sample(feat, rel_img_points)
                    point_feat = point_feat.squeeze(0).transpose(0, 1)
                    point_feats.append(point_feat)
            fine_grained_feats.append(torch.cat(point_feats, dim=0))
        return torch.cat(fine_grained_feats, dim=1)

    def _mask_point_forward_test(self, x, rois, label_pred, mask_pred,
                                 img_metas):
        """Mask refining process with point head in testing."""
        refined_mask_pred = mask_pred.clone()
        for subdivision_step in range(self.test_cfg.subdivision_steps):
            refined_mask_pred = F.interpolate(
                refined_mask_pred,
                scale_factor=self.test_cfg.scale_factor,
                mode='bilinear',
                align_corners=False)
            # If `subdivision_num_points` is larger or equal to the
            # resolution of the next step, then we can skip this step
            num_rois, channels, mask_height, mask_width = \
                refined_mask_pred.shape
            if (self.test_cfg.subdivision_num_points >=
                    self.test_cfg.scale_factor**2 * mask_height * mask_width
                    and
                    subdivision_step < self.test_cfg.subdivision_steps - 1):
                continue
            point_indices, rel_roi_points = \
                self.point_head.get_roi_rel_points_test(
                    refined_mask_pred, label_pred, cfg=self.test_cfg)
            fine_grained_point_feats = self._get_fine_grained_point_feats(
                x, rois, rel_roi_points, img_metas)
            coarse_point_feats = point_sample(mask_pred, rel_roi_points)
            mask_point_pred = self.point_head(fine_grained_point_feats,
                                              coarse_point_feats)

            point_indices = point_indices.unsqueeze(1).expand(-1, channels, -1)
            refined_mask_pred = refined_mask_pred.reshape(
                num_rois, channels, mask_height * mask_width)
            refined_mask_pred = refined_mask_pred.scatter_(
                2, point_indices, mask_point_pred)
            refined_mask_pred = refined_mask_pred.view(num_rois, channels,
                                                       mask_height, mask_width)

        return refined_mask_pred

    def simple_test(self,
                    x,
                    det_bboxes,
                    det_labels,
                    img_metas,
                    rescale=False):
        """Test without augmentation."""
        segm_results = self.simple_test_mask(
            x, img_metas, det_bboxes, det_labels, rescale=rescale)
        return segm_results

    def simple_test_mask(self,
                         x,
                         img_metas,
                         det_bboxes,
                         det_labels,
                         rescale=False):
        """Obtain mask prediction without augmentation."""
        # image shape of the first image in the batch (only one)
        ori_shape = img_metas[0]['ori_shape']
        scale_factor = img_metas[0]['scale_factor']
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            # if det_bboxes is rescaled to the original image size, we need to
            # rescale it back to the testing scale to obtain RoIs.
            if rescale and not isinstance(scale_factor, float):
                scale_factor = det_bboxes.new_tensor(scale_factor)
            _bboxes = (
                det_bboxes[:, :4] * scale_factor if rescale else det_bboxes)
            mask_rois = bbox2roi([_bboxes])
            mask_results = self._mask_forward(x[0], mask_rois)
            mask_results['mask_pred'] = self._mask_point_forward_test(
                x, mask_rois, det_labels, mask_results['mask_pred'], img_metas)
            segm_result = self.mask_head.get_seg_masks(
                mask_results['mask_pred'], _bboxes, det_labels, self.test_cfg,
                ori_shape, scale_factor, rescale)
        return segm_result

    def aug_test_mask(self, feats, img_metas, det_bboxes, det_labels):
        """Test for mask head with test time augmentation."""
        if det_bboxes.shape[0] == 0:
            segm_result = [[] for _ in range(self.mask_head.num_classes)]
        else:
            aug_masks = []
            for x, img_meta in zip(feats, img_metas):
                img_shape = img_meta[0]['img_shape']
                scale_factor = img_meta[0]['scale_factor']
                flip = img_meta[0]['flip']
                _bboxes = bbox_mapping(det_bboxes[:, :4], img_shape,
                                       scale_factor, flip)
                mask_rois = bbox2roi([_bboxes])
                mask_results = self._mask_forward(x, mask_rois)
                mask_results['mask_pred'] = self._mask_point_forward_test(
                    x, mask_rois, det_labels, mask_results['mask_pred'],
                    img_metas)
                # convert to numpy array to save memory
                aug_masks.append(
                    mask_results['mask_pred'].sigmoid().cpu().numpy())
            merged_masks = merge_aug_masks(aug_masks, img_metas, self.test_cfg)

            ori_shape = img_metas[0][0]['ori_shape']
            segm_result = self.mask_head.get_seg_masks(
                merged_masks,
                det_bboxes,
                det_labels,
                self.test_cfg,
                ori_shape,
                scale_factor=1.0,
                rescale=False)
        return segm_result
