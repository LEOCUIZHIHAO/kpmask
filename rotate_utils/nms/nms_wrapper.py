import numpy as np
import torch
import cv2

from . import nms_cpu, nms_cuda


def nms(dets, iou_thr, device_id=None):
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
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms , dets_th[:,:5] to make sure the rotation
    if dets_th.shape[0] == 0:
        inds = dets_th[:,:5].new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:
            inds = nms_cuda.nms(dets_th[:,:5], iou_thr)
        else:
            inds = nms_cpu.nms(dets_th[:,:5], iou_thr)

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds


def soft_nms(dets, iou_thr, method='linear', sigma=0.5, min_score=1e-3):
    """Dispatch to only CPU Soft NMS implementations.

    The input can be either a torch tensor or numpy array.
    The returned type will always be the same as inputs.

    Arguments:
        dets (torch.Tensor or np.ndarray): bboxes with scores.
        iou_thr (float): IoU threshold for Soft NMS.
        method (str): either 'linear' or 'gaussian'
        sigma (float): hyperparameter for gaussian method
        min_score (float): score filter threshold

    Returns:
        tuple: new det bboxes and indice, which is always the same
        data type as the input.

    Example:
        >>> dets = np.array([[4., 3., 5., 3., 0.9],
        >>>                  [4., 3., 5., 4., 0.9],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.5],
        >>>                  [3., 1., 3., 1., 0.4],
        >>>                  [3., 1., 3., 1., 0.0]], dtype=np.float32)
        >>> iou_thr = 0.7
        >>> new_dets, inds = soft_nms(dets, iou_thr, sigma=0.5)
        >>> assert len(inds) == len(new_dets) == 3
    """
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_tensor = True
        dets_t = dets.detach().cpu()
    elif isinstance(dets, np.ndarray):
        is_tensor = False
        dets_t = torch.from_numpy(dets)
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    method_codes = {'linear': 1, 'gaussian': 2}
    if method not in method_codes:
        raise ValueError('Invalid method for SoftNMS: {}'.format(method))
    results = nms_cpu.soft_nms(dets_t, iou_thr, method_codes[method], sigma,
                               min_score)

    new_dets = results[:, :5]
    inds = results[:, 5]

    if is_tensor:
        return new_dets.to(
            device=dets.device, dtype=dets.dtype), inds.to(
                device=dets.device, dtype=torch.long)
    else:
        return new_dets.numpy().astype(dets.dtype), inds.numpy().astype(
            np.int64)


def normal_nms(dets, iou_thr, device_id=None, EPSILON=1e-5):

    if isinstance(dets, torch.Tensor):
        dets_th = dets.cpu().numpy().copy()
        dets_score = dets_th[:,-1]

    #opencv ([-90 - 0 ] in degree)
    # WARNING:  convert predicted [top_left, bottom_right] to [center w h]
    d_theta = np.degrees(dets_th[:,4:5])
    dets_th = dets_th[:, :4]

    keep = []
    order = dets_score.argsort()[::-1]
    num = dets_th.shape[0]
    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        #print("box idx_", _i)
        i = order[_i]

        if suppressed[i] == 1:
           continue
        #print("_i", _i)
        keep.append(i)
        area_r1 = (dets_th[i, 2] - dets_th[i, 0]) * (dets_th[i, 3] - dets_th[i, 1]) + 1
        #print("************************")
        for _j in range(_i + 1, num):
            #print("compare with box idx_", _j)
            j = order[_j]

            if suppressed[i] == 1:
               continue

            area_r2 = (dets_th[j, 2] - dets_th[j, 0]) * (dets_th[j, 3] - dets_th[j, 1]) + 1

            lt = np.maximum(dets_th[j, :2], dets_th[i, :2])  # [rows, 2]
            rb = np.minimum(dets_th[j, 2:4], dets_th[i, 2:4])  # [rows, 2]

            wh = (rb - lt + 1).clip(min=0)  # [rows, 2] #clip make sure the intersection 0 -> iou=0
            int_area = wh[0] * wh[1] + 1

            angle_diff = np.abs(d_theta[i] - d_theta[j])
            inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + EPSILON)
            #print("inter : ", inter)

            if inter >= iou_thr:
                suppressed[j] = 1

    return dets[keep, :], keep


def rotate_nms(dets, iou_thr, device_id=None, EPSILON=1e-5):

    if isinstance(dets, torch.Tensor):
        dets_th = dets.cpu().numpy().copy()
        dets_score = dets_th[:,-1]

    #opencv ([-90 - 0 ] in degree)
    # WARNING:  convert predicted [top_left, bottom_right] to [center w h]
    d_w = dets_th[:, 2:3] - dets_th[:, 0:1]
    d_h = dets_th[:, 3:4] - dets_th[:, 1:2]
    d_ctr_x = dets_th[:, 0:1] + d_w * 0.5
    d_ctr_y = dets_th[:, 1:2] + d_h * 0.5
    d_theta = np.degrees(dets_th[:,4:5])
    dets_th = np.concatenate([d_ctr_x, d_ctr_y, d_w, d_h, d_theta], axis=-1)

    keep = []
    order = dets_score.argsort()[::-1]
    num = dets_th.shape[0]
    suppressed = np.zeros((num), dtype=np.int)

    for _i in range(num):
        #print("box idx_", _i)
        i = order[_i]

        if suppressed[i] == 1:
           continue
        #print("_i", _i)
        keep.append(i)
        r1 = ((dets_th[i, 0], dets_th[i, 1]), (dets_th[i, 2], dets_th[i, 3]), dets_th[i, 4])
        area_r1 = dets_th[i, 2] * dets_th[i, 3]
        for _j in range(_i + 1, num):
            #print("compare with box idx_", _j)
            j = order[_j]

            if suppressed[i] == 1:
               continue

            r2 = ((dets_th[j, 0], dets_th[j, 1]), (dets_th[j, 2], dets_th[j, 3]), dets_th[j, 4])
            area_r2 = dets_th[j, 2] * dets_th[j, 3]
            inter = 0.0

            try:
                ret_type, int_pts = cv2.rotatedRectangleIntersection(r1, r2)
                #print("ret_type : ", ret_type)
                if ret_type == 0:
                    #print("ret_type : ", ret_type)
                    continue

                if int_pts is not None:
                    order_pts = cv2.convexHull(int_pts, returnPoints=True)

                    int_area = cv2.contourArea(order_pts)

                    #print("int_area : ", int_area)
                    #print("union : ", (area_r1 + area_r2 - int_area + EPSILON))
                    inter = int_area * 1.0 / (area_r1 + area_r2 - int_area + EPSILON)
                    #print("inter : ", inter)
            except:
                """
                  cv2.error: /io/opencv/modules/imgproc/src/intersection.cpp:247:
                  error: (-215) intersection.size() <= 8 in function rotatedRectangleIntersection
                """
                # print(r1)
                # print(r2)
                raise
                inter = 0.9999

            if inter >= iou_thr:
                suppressed[j] = 1

    return dets[keep, :], keep


## TODO: Not working
def rotate_nms_gpu(dets, iou_thr, device_id=None):
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

    assert dets.shape[-1] == 6
    # convert dets (tensor or numpy array) to tensor
    if isinstance(dets, torch.Tensor):
        is_numpy = False
        dets_th = dets[:,:5] #ignore score
    elif isinstance(dets, np.ndarray):
        is_numpy = True
        device = 'cpu' if device_id is None else 'cuda:{}'.format(device_id)
        dets_th = torch.from_numpy(dets).to(device)[:,:5]#ignore score
    else:
        raise TypeError(
            'dets must be either a Tensor or numpy array, but got {}'.format(
                type(dets)))

    # execute cpu or cuda nms
    if dets_th.shape[0] == 0:
        inds = dets_th.new_zeros(0, dtype=torch.long)
    else:
        if dets_th.is_cuda:

            dets_th = dets_th.cpu().numpy()

            # WARNING:  convert predicted [top_left, bottom_right] to [center w h]
            d_w = dets_th[:, 2:3] - dets_th[:, 0:1]
            d_h = dets_th[:, 3:4] - dets_th[:, 1:2]
            d_ctr_x = dets_th[:, 0:1] + d_w * 0.5
            d_ctr_y = dets_th[:, 1:2] + d_h * 0.5
            d_theta = dets_th[:,4:5]
            dets_th = np.concatenate([d_ctr_x, d_ctr_y, d_w, d_h, d_theta], axis=-1)

            for idx in range(dets_th.shape[0]):
                rot_box = ((dets_th[idx, 0], dets_th[idx, 1]),
                           (dets_th[idx, 2], dets_th[idx, 3]),
                           np.degrees(dets_th[idx, 4]))

                corner_pt = cv2.boxPoints(rot_box)
                rot_box_edge = [corner_pt[1] - corner_pt[0], corner_pt[3] - corner_pt[0]]
                np.linalg.norm(rot_box_edge)
                wh_idx = sorted((np.linalg.norm(edge), idx) for idx, edge in enumerate(rot_box_edge))
                long_axis = rot_box_edge[wh_idx[1][1]]
                theta = np.arctan2(long_axis[1], long_axis[0])
                if theta > np.pi / 2.0:
                    theta -= np.pi
                if theta < -np.pi / 2.0:
                    theta += np.pi

                dets_th[idx, 2] = float(wh_idx[1][0])
                dets_th[idx, 3] = float(wh_idx[0][0])
                dets_th[idx, 4] = float(-theta * 180 / np.pi)

            dets_th = torch.from_numpy(dets_th).to('cuda')
            # [top_left, bottom_right] as input rotate along box center
            inds = nms_cuda.rotate_nms(dets_th, iou_thr)

        else:
            raise ValueError('RoateNMS Not Support CPU')

    if is_numpy:
        inds = inds.cpu().numpy()
    return dets[inds, :], inds
