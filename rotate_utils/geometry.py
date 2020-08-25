import torch
import numba
import numpy as np
from mmdet.ops.rbbox_overlaps.rbbox_overlaps_wrapper import rotate_iou_cuda


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4) in <x1, y1, x2, y2> format.
        bboxes2 (Tensor): shape (n, 4) in <x1, y1, x2, y2> format.
            If is_aligned is ``True``, then m and n must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)

    Example:
        >>> bboxes1 = torch.FloatTensor([
        >>>     [0, 0, 10, 10],
        >>>     [10, 10, 20, 20],
        >>>     [32, 32, 38, 42],
        >>> ])
        >>> bboxes2 = torch.FloatTensor([
        >>>     [0, 0, 10, 20],
        >>>     [0, 10, 10, 19],
        >>>     [10, 10, 20, 20],
        >>> ])
        >>> bbox_overlaps(bboxes1, bboxes2)
        tensor([[0.5238, 0.0500, 0.0041],
                [0.0323, 0.0452, 1.0000],
                [0.0000, 0.0000, 0.0000]])

    Example:
        >>> empty = torch.FloatTensor([])
        >>> nonempty = torch.FloatTensor([
        >>>     [0, 0, 10, 9],
        >>> ])
        >>> assert tuple(bbox_overlaps(empty, nonempty).shape) == (0, 1)
        >>> assert tuple(bbox_overlaps(nonempty, empty).shape) == (1, 0)
        >>> assert tuple(bbox_overlaps(empty, empty).shape) == (0, 0)
    """
    assert mode in ['iou', 'iof', 'skewiou']
    rows = bboxes1.size(0)
    cols = bboxes2.size(0)

    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:4], bboxes2[:, 2:4])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1

    else:
        if mode == 'skewiou':
            # top_right, bottom_right 2 (center, w, h)
            bboxes1_ctr_x = (bboxes1[:, 0:1] + bboxes1[:,2:3]) * 0.5 + 1
            bboxes1_ctr_y = (bboxes1[:, 1:2] + bboxes1[:,3:4]) * 0.5 + 1
            bboxes1_w = bboxes1[:, 2:3] - bboxes1[:,0:1] + 1
            bboxes1_h = bboxes1[:, 3:4] - bboxes1[:,1:2] + 1
            bboxes1_theta = bboxes1[:,4:5] * 180 / 3.14 #radius to degree

            bboxes2_ctr_x = (bboxes2[:, 0:1] + bboxes2[:,2:3]) * 0.5 + 1
            bboxes2_ctr_y = (bboxes2[:, 1:2] + bboxes2[:,3:4]) * 0.5 + 1
            bboxes2_w = bboxes2[:, 2:3] - bboxes2[:,0:1] + 1
            bboxes2_h = bboxes2[:, 3:4] - bboxes2[:,1:2] + 1
            bboxes2_theta = bboxes2[:, 4:5] * 180 / 3.14

            bboxes1 = torch.cat([bboxes1_ctr_x, bboxes1_ctr_y, bboxes1_w, bboxes1_h, bboxes1_theta], dim=-1)
            bboxes2 = torch.cat([bboxes2_ctr_x, bboxes2_ctr_y, bboxes2_w, bboxes2_h, bboxes2_theta], dim=-1)

            ious = rotate_iou_cuda(bboxes1, bboxes2, device_id=torch.cuda.current_device())

        else:
            lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
            rb = torch.min(bboxes1[:, None, 2:4], bboxes2[:, 2:4])  # [rows, cols, 2]

            wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
            overlap = wh[:, :, 0] * wh[:, :, 1]
            area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
                bboxes1[:, 3] - bboxes1[:, 1] + 1)

            if mode == 'iou':
                area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                    bboxes2[:, 3] - bboxes2[:, 1] + 1)
                ious = overlap / (area1[:, None] + area2 - overlap)
            else:
                ious = overlap / (area1[:, None])

    return ious
