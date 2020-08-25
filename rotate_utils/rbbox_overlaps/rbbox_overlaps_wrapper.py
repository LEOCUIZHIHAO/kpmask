import numpy as np
import torch
import cv2

from . import rbbox_overlaps


def rotate_iou_cuda(boxes, query_boxes, device_id=None):
    """Dispatch to either CPU or GPU NMS implementations.

    The input can be either a torch tensor or numpy array. GPU NMS will be used
    if the input is a gpu tensor or device_id is specified, otherwise CPU NMS
    will be used. The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for NMS.
        device_id (int, optional): when `dets` is a numpy array, if `device_id`
            is None, then cpu nms is used, otherwise gpu_nms will be used.

    Returns:
        tuple: kept bboxes and indice, which is always the same data type as
            the input.

    Example:
        >>> dets = np.array([[49.1, 32.4, 51.0, 35.9, 0.9],
        >>>                  [49.3, 32.9, 51.0, 35.3, 0.9],
        >>>                  [49.2, 31.8, 51.0, 35.4, 0.5],
        >>>                  [35.1, 11.5, 39.1, 15.7, 0.5],
        >>>                  [35.6, 11.8, 39.3, 14.2, 0.5],
        >>>                  [35.3, 11.5, 39.9, 14.5, 0.4],
        >>>                  [35.2, 11.7, 39.7, 15.7, 0.3]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> suppressed, inds = nms(dets, iou_thr)
        >>> assert len(inds) == len(suppressed) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(boxes, torch.Tensor) and isinstance(query_boxes, torch.Tensor):
        pass
    else:
        raise TypeError(
            'boxes and query_boxes must be either a Tensor , but got {} and {}'.format(
                type(boxes), type(query_boxes)))

    if boxes.shape[0] == 0:
        ious = boxes.new_zeros(0, dtype=torch.long)
    else:
        ious = rbbox_overlaps.rbbox_iou(boxes, query_boxes, device_id)

    return ious
