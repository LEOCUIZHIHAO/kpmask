import torch
import torch.nn as nn
from ..builder import HEADS
from ..builder import build_loss

from mmcv.cnn import (ConvModule, bias_init_with_prob, normal_init)
from mmdet.core import (multi_apply, force_fp32, ctdet_decode, keypoint_box_nms)

@HEADS.register_module
class KeyPointHead(nn.Module):
    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels,
                 stacked_convs=2,
                 deconv_method=None,
                 loss_cls=dict(type='KeyPointFocalLoss'),
                 loss_bbox=dict(type='KeyPointRegL1Loss'),
                 loss_offset=dict(type='KeyPointRegL1Loss'),
                 loss_iou=dict(type='CIoULoss'),
                 conv_cfg=None,
                 norm_cfg=None, #dict(type='GN', num_groups=32, requires_grad=True),
                 train_cfg=None,
                 test_cfg=None):
        super(KeyPointHead, self).__init__()
        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.down_ratio=4 ## WARNING:  hard_code for TTA
        self.stacked_convs = stacked_convs
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_offset = build_loss(loss_offset)
        self.loss_iou = build_loss(loss_iou)
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.fp16_enabled = False
        self.multi_stack=False
        if len(self.test_cfg['stack_out_level']) > 1: self.multi_stack=True

    def init_weights(self):
        pass

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x):
        hm_pred = x['hm']
        wh_pred = x['wh']
        reg_pred = x['reg']
        return hm_pred, wh_pred, reg_pred

    @force_fp32(apply_to=('hard_bboxes_pred', 'hard_gt_bbox'))
    def ohem_loss(self,
                 hard_bboxes_pred,
                 hard_gt_bbox,
                 hard_hm_pred,
                 hard_gt_hm,
                 ):
        batch=len(hard_bboxes_pred)
        losses_iou = 0
        losses_hard_cls = 0
        for i in range(batch):
            losses = self.loss_iou(hard_bboxes_pred[i].float(), hard_gt_bbox[i].float())
            losses_iou += losses

            # sort the loss diff according to the categories
            _, K, cat = hard_gt_hm[i].size()
            class_diff = torch.clamp(hard_gt_hm[i] - hard_hm_pred[i], 0)
            weight = torch.gt(class_diff, 0.3).float() ## WARNING:  hard_code
            class_diff = class_diff * weight
            class_diff = class_diff.permute(2, 0, 1).contiguous()
            topk_hard_score, topk_hard_ind = torch.topk(class_diff.view(cat, -1), K, sorted=False) # ([cat, K])
            simple_dif_sum = torch.mean(topk_hard_score) # use hard thresh or use sum
            losses_hard_cls +=simple_dif_sum

        return dict(loss_iou=losses_iou, loss_hard_cls=losses_hard_cls)

    def loss_single(self, hm_pred, wh_pred, reg_pred, gt_hm, gt_wh, gt_reg, gt_inds, gt_reg_mask):
        weight=None
        losses_cls = self.loss_cls(self._sigmoid(hm_pred), gt_hm, weight) / self.stacked_convs
        losses_bbox = self.loss_bbox(wh_pred, gt_reg_mask, gt_inds, gt_wh) / self.stacked_convs
        losses_offset = self.loss_offset(reg_pred, gt_reg_mask, gt_inds, gt_reg) / self.stacked_convs
        return losses_cls, losses_bbox, losses_offset

    @force_fp32(apply_to=('hm_pred', 'wh_pred', 'reg_pred'))
    def loss(self,
             hm_pred,
             wh_pred,
             reg_pred,
             gt_hm,
             gt_wh,
             gt_reg,
             gt_inds,
             gt_reg_mask,
             gt_bboxes_ignore=None):
        assert len(hm_pred) == len(wh_pred) == len(reg_pred)
        losses_cls, losses_bbox, losses_offset = multi_apply(
                                                    self.loss_single,
                                                    hm_pred,
                                                    wh_pred,
                                                    reg_pred,
                                                    gt_hm,
                                                    gt_wh,
                                                    gt_reg,
                                                    gt_inds,
                                                    gt_reg_mask,
                                                    )
        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_offset=losses_offset)

    @force_fp32(apply_to=('hm_pred', 'wh_pred', 'reg_pred'))
    def get_bboxes(self,
                   hm_pred,
                   wh_pred,
                   reg_pred,
                   img_metas,
                   down_ratios=[(4,4)],
                   rescale=None):
        assert len(hm_pred) == len(wh_pred) == len(reg_pred)
        num_levels = len(hm_pred)
        keep_objs = self.test_cfg['K']

        ## WARNING:  sigmoid, stack head level
        if self.multi_stack:
            hm_pred_list = [hm_pred[i].detach().sigmoid() for i in range(num_levels)]
            wh_pred_list = [wh_pred[i].detach() for i in range(num_levels)]
            reg_pred_list = [reg_pred[i].detach() for i in range(num_levels)]

            hm_pred_list = torch.stack(hm_pred_list).mean(dim=0)
            wh_pred_list = torch.stack(wh_pred_list).mean(dim=0)
            reg_pred_list = torch.stack(reg_pred_list).mean(dim=0)

        else:
            # stack_level = self.test_cfg['stack_out_level'][0]
            stack_level = self.test_cfg['stack_out_level'][0]
            hm_pred_list = hm_pred[stack_level].detach().sigmoid()
            wh_pred_list = wh_pred[stack_level].detach()
            reg_pred_list = reg_pred[stack_level].detach()

        bboxes, cls_score, clses = ctdet_decode(hm_pred_list,
                                                wh_pred_list,
                                                reg=reg_pred_list,
                                                K=keep_objs)
        down_ratio = down_ratios[stack_level]
        bboxes = bboxes * down_ratio[0] ## WARNING: 128x128 -> 512x512
        det_bboxes = torch.cat([bboxes, cls_score], -1)
        # print("[debug] down_ratio", down_ratio)

        det_bboxes = det_bboxes.view(keep_objs, 5)
        cls_score = cls_score.view(keep_objs)
        clses = clses.view(keep_objs)

        if self.test_cfg.score_thr > 0:
            inds = cls_score>self.test_cfg.score_thr
            det_bboxes = det_bboxes[inds]
            clses = clses[inds]
            if self.test_cfg.nms is not None:
                det_bboxes, clses = keypoint_box_nms(det_bboxes[:,:4], cls_score[inds], clses, self.test_cfg.nms)

        if rescale:
            scale_factor = img_metas[0]['scale_factor']
            det_bboxes[:,:4] /= det_bboxes[:,:4].new_tensor(scale_factor) ## WARNING: 512*512 -> original pic size

        return det_bboxes, clses

    def aug_test_bboxes(self, feats, img_metas, rescale):


        assert self.multi_stack==False, "Not Support Multi Stack"

        stack_level = self.test_cfg['stack_out_level'][0]

        aug_hm = []
        aug_wh = []
        aug_reg = []

        aug_hm_c = [i for i in range(self.stacked_convs)]
        aug_wh_c = [i for i in range(self.stacked_convs)]
        aug_reg_c = [i for i in range(self.stacked_convs)]

        for x, img_meta in zip(feats, img_metas):
            # if not self.multi_stack:
            hm_pred, wh_pred, reg_pred = self.forward(x)
            #hourglass stacks merge
            if img_meta[0]['flip']:
                w = int(img_meta[0]['img_shape'][1] / self.down_ratio) #for horizontal flip aug
                hm_pred[stack_level][..., :w] = torch.flip(hm_pred[stack_level][..., :w], [3])
                wh_pred[stack_level][..., :w] = torch.flip(wh_pred[stack_level][..., :w], [3])
                reg_pred[stack_level][..., :w] = torch.flip(reg_pred[stack_level][..., :w], [3])
            aug_hm.append(hm_pred[stack_level])
            aug_wh.append(wh_pred[stack_level])
            aug_reg.append(reg_pred[stack_level])

        aug_hm_c[stack_level]=torch.stack(aug_hm).mean(dim=0)
        aug_wh_c[stack_level]=torch.stack(aug_wh).mean(dim=0)
        aug_reg_c[stack_level]=torch.stack(aug_reg).mean(dim=0)

        outs = (aug_hm_c, aug_wh_c, aug_reg_c)

        bboxes, labels = self.get_bboxes(
                        *outs,
                        img_meta,
                        rescale=rescale)

        return bboxes, labels

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid_(), min=1e-4, max=1-1e-4)
        return y
