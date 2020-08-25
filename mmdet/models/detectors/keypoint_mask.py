import numpy as np
import torch

import math
from torch import nn
from ..builder import DETECTORS, build_backbone, build_head, build_neck
from mmdet.ops import (draw_umich_gaussian, draw_dense_reg, gaussian_radius)
from mmdet.core import bbox2result, transpose_and_gather_feat
from .base import BaseDetector

def check_fm_size(featmap_sizes):
    for i in range(len(featmap_sizes)-1):
        if featmap_sizes[i][:]!=featmap_sizes[i+1][:]:
            return False
    return True

def _repeat(_list, repeat_times):
    for _ in range(repeat_times):
        _list.append(_list[0].copy())
    return _list

def get_down_ratios(img, outs, stack_nums):
    down_ratios=[]
    for i in range(stack_nums):
        out_fm_h, out_fm_w = img.shape[2:4]
        height, width = outs[0][i].shape[2:4]
        down_ratios.append((int(out_fm_h/height), int(out_fm_w/width)))
    return down_ratios

@DETECTORS.register_module()
class Keypoint_mask(BaseDetector):

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 rpn_head=None,
                 roi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(Keypoint_mask, self).__init__()
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if bbox_head is not None:
            bbox_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.bbox_head = build_head(bbox_head)
            self.box_head_deconv_method = bbox_head.deconv_method
            self.box_head_stack_nums = bbox_head.stacked_convs

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)

        if roi_head is not None:
            roi_head.update(train_cfg=train_cfg, test_cfg=test_cfg)
            self.roi_head = build_head(roi_head)

        if train_cfg is not None:
            self.mask_size = train_cfg.mask_size
            self.dynamic_training = train_cfg.dynamic_training
            self.max_objs = train_cfg.max_objs
            self.num_classes = train_cfg.num_classes
            self.opt = train_cfg.opt
            self.debug = train_cfg.gt_debug

        self.init_weights(pretrained=pretrained)

    @property
    def with_rpn(self):
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self):
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def init_weights(self, pretrained=None):
        super(Keypoint_mask, self).init_weights(pretrained)
        self.backbone.init_weights(pretrained=pretrained)
        if self.with_neck:
            if isinstance(self.neck, nn.Sequential):
                for m in self.neck:
                    m.init_weights()
            else:
                self.neck.init_weights()
        self.bbox_head.init_weights()
        if self.with_rpn:
            self.rpn_head.init_weights()
        if self.with_roi_head:
            self.roi_head.init_weights(pretrained)

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck
        """
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def aug_test(self, imgs, img_metas, rescale=False):
        x = self.extract_feats(imgs)
        det_bboxes, det_labels = self.bbox_head.aug_test_bboxes(x, img_metas, rescale)

        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        # if not self.with_mask:
        return bbox_results

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs,)
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs,)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        """
        Args:
            img (list[Tensor]): list of tensors of shape (1, C, H, W).
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has:
                'img_shape', 'scale_factor', 'flip', and my also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

             **kwargs: specific to concrete implementation
        """
        if self.debug:
            from mmdet.core import tensor2imgs
            import mmcv
            _img = tensor2imgs(img)
            _gt_labels = gt_labels[0].cpu().numpy()
            print("gt_label : ", _gt_labels)
            print("img shape : ", _img[0].shape)
            print("num of gt_bboxes : ", gt_bboxes[0].shape)
            if gt_masks is not None:
                _gt_masks = gt_masks[0].to_ndarray().astype(np.bool)
                inds = gt_bboxes[0].shape[0]
                for i in range(inds):
                    color_mask = np.random.randint(
                        0, 256, (1, 3), dtype=np.uint8)
                    mask = _gt_masks[i]
                    _img[0][mask] = _img[0][mask] * 0.5 + color_mask * 0.5
            mmcv.imshow_det_bboxes(_img[0], gt_bboxes[0].cpu().numpy(), _gt_labels, bbox_color='green', thickness=2,
                                   show_confidence=False, show_rotate=False)

        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if isinstance(x[0], dict): x = [x[i]['fm'] for i in range(len(x))]
        losses = dict()

        featmap_sizes = [featmap.size()[-2:] for featmap in x]
        down_ratios = get_down_ratios(img, outs, self.box_head_stack_nums)
        # debug_featmap_sizes = [print("featmap", featmap.shape) for featmap in x]
        # print("[debug] down_ratios", down_ratios)

        top_down_heatmaps = self.generate_gt(gt_bboxes, gt_labels, gt_masks, featmap_sizes, down_ratios)
        gt_hm, gt_wh, gt_reg, gt_inds, gt_reg_mask, gt_label, gt_masks_roi, rois, gt_masks_bitmap = self.numpy_to_torch(
            top_down_heatmaps)  # (gt_hm, gt_wh, gt_reg, gt_inds, gt_reg_mask, gt_labels, mask_targets, batch_rois)
        # box losses
        loss_inputs = outs + (gt_hm, gt_wh, gt_reg, gt_inds, gt_reg_mask)
        box_losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        losses.update(box_losses)

        if self.dynamic_training is not None:
            # refine the gt boxes
            hard_bboxes_pred, hard_gt_bbox, hard_hm_pred, hard_gt_hm = self.get_hard_samples(*outs, gt_hm,
                                                                                    gt_inds, gt_bboxes,
                                                                                    self.dynamic_training.refine_level,
                                                                                    down_ratios) ## WARNING:  hard code 1 is for level HG
            hard_losses = self.bbox_head.ohem_loss(hard_bboxes_pred, hard_gt_bbox,
                                                    hard_hm_pred, hard_gt_hm) #hg last layer
            losses.update(hard_losses)

        roi_losses = self.roi_head.forward_train(x, img_metas,
                                                 rois, gt_label,
                                                 gt_bboxes_ignore,
                                                 gt_masks_roi=gt_masks_roi,
                                                 gt_masks_bitmap=gt_masks_bitmap[0],
                                                 )
        losses.update(roi_losses)
        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = await self.async_test_rpn(x, img_meta)
        else:
            proposal_list = proposals
        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        x = self.extract_feat(img)
        outs = self.bbox_head(x)
        if isinstance(x[0], dict): x = [x[i]['fm'] for i in range(len(x))]
        down_ratios = get_down_ratios(img, outs, self.box_head_stack_nums)
        det_bboxes, det_labels = self.bbox_head.get_bboxes(*outs, img_metas, down_ratios, rescale)
        bbox_results = bbox2result(det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        if not self.with_mask:
            return bbox_results
        else:
            segm_results = self.roi_head.simple_test(
                x, det_bboxes, det_labels, img_metas, rescale=rescale)
            return bbox_results, segm_results

    def generate_gt(self, gt_bboxes, gt_labels, gt_masks, featmap_sizes, down_ratios):
        gt_nums = len(featmap_sizes)
        top_down_heatmaps = []
        fm_size_match = check_fm_size(featmap_sizes) #Input [B, C, H, W]
        if fm_size_match: del featmap_sizes[1:] #remove redundant fm size
        if self.box_head_deconv_method is not None: del featmap_sizes[1:] #remove redundant fm size
        if self.debug: print("[debug] featmap_sizes", featmap_sizes)

        for i, featmap_size in enumerate(featmap_sizes):
            out_fm_h = featmap_size[0]
            out_fm_w = featmap_size[1]
            down_ratio = down_ratios[i]
            heatmaps = dict()
            for gt_box, gt_label, gt_mask in zip(gt_bboxes, gt_labels, gt_masks):
                heatmap = self.get_targets(gt_box, gt_label, gt_mask, \
                                                (out_fm_h, out_fm_w), down_ratio)
                for key in heatmap.keys():
                    if key in heatmaps:
                        heatmaps[key].append(heatmap[key])
                    else:
                        heatmaps[key] = [heatmap[key]]
            top_down_heatmaps.append(heatmaps)

        if fm_size_match: top_down_heatmaps = _repeat(top_down_heatmaps, gt_nums-1)

        return top_down_heatmaps

    def get_targets(self, gt_bboxes, gt_labels, gt_masks, featmap_size, down_ratio):
        assert featmap_size[0] == featmap_size[1] , "Not support FM with differnt width and hight"
        featmap_size = featmap_size[0]
        down_ratio = down_ratio[0]

        hm = np.zeros((self.num_classes, featmap_size, featmap_size), dtype=np.float32)
        wh = np.zeros((self.max_objs, 2), dtype=np.float32)
        dense_wh = np.zeros((2, featmap_size, featmap_size), dtype=np.float32)
        reg = np.zeros((self.max_objs, 2), dtype=np.float32)
        ind = np.zeros((self.max_objs), dtype=np.int64)
        reg_mask = np.zeros((self.max_objs), dtype=np.uint8)
        cat_spec_wh = np.zeros((self.max_objs, self.num_classes * 2), dtype=np.float32)
        cat_spec_mask = np.zeros((self.max_objs, self.num_classes * 2), dtype=np.uint8)

        gt_bboxes = gt_bboxes.cpu().numpy()
        gt_labels = gt_labels.cpu().numpy()
        _gt_bboxes = gt_bboxes[:, :4] ## WARNING: Careful for rotation
        keep_inds = []

        assert gt_bboxes.shape[0] == gt_labels.shape[0] == len(gt_masks)

        for k in range(gt_bboxes.shape[0]):
            bbox = _gt_bboxes[k] / down_ratio
            cls_id = gt_labels[k]
            h, w = bbox[3] - bbox[1], bbox[2] - bbox[0]
            if h > 0 and w > 0:
                keep_inds.append(k)
                radius, max_radius = gaussian_radius((math.ceil(h), math.ceil(w)))
                radius = max(0, int(radius))
                ct = np.array(
                    [(bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2], dtype=np.float32)
                ct_int = ct.astype(np.int32)
                draw_umich_gaussian(hm[cls_id], ct, radius)
                wh[k] = 1. * w, 1. * h
                ind[k] = ct_int[1] * featmap_size + ct_int[0]
                reg[k] = ct - ct_int
                reg_mask[k] = 1
                cat_spec_wh[k, cls_id * 2: cls_id * 2 + 2] = wh[k]
                cat_spec_mask[k, cls_id * 2: cls_id * 2 + 2] = 1
                if self.opt['dense_wh']: draw_dense_reg(dense_wh, hm.max(axis=0), ct_int, wh[k], radius)

                if False:
                    import mmcv
                    l = np.array([cls_id])
                    mmcv.imshow_det_bboxes(hm[cls_id] * 255, bbox.reshape(1,-1), l, show_confidence=False)

        # select the box w & h greater than 0
        _gt_bboxes = _gt_bboxes[keep_inds]
        gt_masks = gt_masks[keep_inds]
        gt_labels = gt_labels[keep_inds]

        mask = gt_masks.crop_and_resize(_gt_bboxes, (self.mask_size, self.mask_size), \
                                        inds=np.arange(_gt_bboxes.shape[0])).to_ndarray().astype(np.float32)

        ret = {'hm': hm, 'reg_mask': reg_mask, 'ind': ind, 'wh': wh, \
               'mask': mask, 'bitmap_mask': gt_masks, 'labels': gt_labels, 'box': _gt_bboxes, 'reg': reg}
        return ret

    def numpy_to_torch(self, top_down_heatmaps):
        _gt_hm=[]
        _gt_wh=[]
        _gt_reg=[]
        _gt_inds=[]
        _gt_reg_mask=[]
        _gt_labels=[]
        _mask_targets=[]
        _batch_rois=[]
        _bitmap_mask_targets=[]
        for heatmaps in top_down_heatmaps:
            gt_hm = torch.from_numpy(np.array(heatmaps['hm'])).cuda()
            gt_wh = torch.from_numpy(np.array(heatmaps['wh'])).cuda() #(B, max_obj, 2)
            gt_reg = torch.from_numpy(np.array(heatmaps['reg'])).cuda() #(B, max_obj, 2)
            gt_inds = torch.from_numpy(np.array(heatmaps['ind'])).cuda()
            gt_reg_mask = torch.from_numpy(np.array(heatmaps['reg_mask'])).cuda()
            gt_labels = torch.from_numpy(np.concatenate(heatmaps['labels'])).cuda()
            batch_size = gt_hm.shape[0]

            batch_rois, mask_targets = [], []
            for batch in range(batch_size): ## WARNING:
                mask_targets.append(torch.from_numpy(heatmaps['mask'][batch]).cuda())
                rois = torch.from_numpy(heatmaps['box'][batch].astype(np.float32)).cuda() # WARNING: shold be 512 x 512 scale, dtype float32
                batch_inds = torch.tensor([batch], dtype=rois.dtype, device=rois.device)
                batch_inds = batch_inds.expand(rois.shape[0], 1)
                rois = torch.cat([batch_inds, rois], 1)
                batch_rois.append(rois)
            mask_targets = torch.cat(mask_targets, 0)
            batch_rois = torch.cat(batch_rois, 0)

            _gt_hm.append(gt_hm)
            _gt_wh.append(gt_wh)
            _gt_reg.append(gt_reg)
            _gt_inds.append(gt_inds)
            _gt_reg_mask.append(gt_reg_mask)
            _gt_labels.append(gt_labels)
            _mask_targets.append(mask_targets)
            _batch_rois.append(batch_rois)
            _bitmap_mask_targets.append(heatmaps['bitmap_mask'])
        return _gt_hm, _gt_wh, _gt_reg, _gt_inds, _gt_reg_mask, _gt_labels, _mask_targets, _batch_rois, _bitmap_mask_targets

    def get_hard_samples(self, hm_pred, wh_pred, reg_pred, gt_hm, gt_inds, gt_bboxes, num_levels, down_ratios):
        #op last layer of HG
        hm_pred = hm_pred[num_levels]
        wh_pred = wh_pred[num_levels]
        reg_pred = reg_pred[num_levels]
        gt_hm = gt_hm[num_levels]
        gt_inds = gt_inds[num_levels]
        down_ratio = down_ratios[num_levels]
        batch, cat, height, width = hm_pred.size()
        hard_bboxes_pred, hard_gt_bbox = [], []
        hard_hm_pred, hard_gt_hm = [], []

        for i in range(batch):
            _gt_hm = gt_hm[i].unsqueeze(0)
            _hm_pred = hm_pred[i].unsqueeze(0)
            _gt_bbox = gt_bboxes[i] / down_ratio[0] ## WARNING: hard code
            _reg_pred = reg_pred[i].unsqueeze(0)
            _wh_pred = wh_pred[i].unsqueeze(0)
            K = gt_bboxes[i].shape[0] #number of gt

            gt_box_inds = gt_inds[i][:K].unsqueeze(0)
            ys = (gt_box_inds / width).int().float()
            xs = (gt_box_inds % width).int().float()

            # decode pred_boxes according to the gt_hm
            _reg_pred = transpose_and_gather_feat(_reg_pred, gt_box_inds)
            _reg_pred = _reg_pred.view(1, K, 2)
            xs = xs.view(1, K, 1) + _reg_pred[:, :, 0:1]
            ys = ys.view(1, K, 1) + _reg_pred[:, :, 1:2]
            _wh_pred = transpose_and_gather_feat(_wh_pred, gt_box_inds)
            _wh_pred = _wh_pred.view(1, K, 2)
            bboxes_pred = torch.cat([xs - _wh_pred[..., 0:1] / 2,
                                ys - _wh_pred[..., 1:2] / 2,
                                xs + _wh_pred[..., 0:1] / 2,
                                ys + _wh_pred[..., 1:2] / 2], dim=2)
            bboxes_pred = bboxes_pred.squeeze(0)
            hard_gt_bbox.append(_gt_bbox)
            hard_bboxes_pred.append(bboxes_pred)

            # hm dynamic_training
            _gt_hm = transpose_and_gather_feat(_gt_hm, gt_box_inds) #([1, K, cat])
            _hm_pred = transpose_and_gather_feat(_hm_pred, gt_box_inds) #([1, K, cat])
            hard_gt_hm.append(_gt_hm)
            hard_hm_pred.append(_hm_pred)

        return hard_bboxes_pred, hard_gt_bbox, hard_hm_pred, hard_gt_hm
