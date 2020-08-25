import torch


class AnchorGenerator(object):
    """
    Examples:
        >>> from mmdet.core import AnchorGenerator
        >>> self = AnchorGenerator(9, [1.], [1.])
        >>> all_anchors = self.grid_anchors((2, 2), device='cpu')
        >>> print(all_anchors)
        tensor([[ 0.,  0.,  8.,  8.],
                [16.,  0., 24.,  8.],
                [ 0., 16.,  8., 24.],
                [16., 16., 24., 24.]])
    """

    def __init__(self, base_size, scales, ratios, use_rotate_box, rotate_range,
                use_anchor_ctr_offset=False, scale_major=True, ctr=None):
        self.base_size = base_size
        self.scales = torch.Tensor(scales)
        self.ratios = torch.Tensor(ratios)
        self.scale_major = scale_major
        self.ctr = ctr
        self.use_rotate_box = use_rotate_box
        self.rotate_range = torch.Tensor(rotate_range)
        self.use_anchor_ctr_offset = use_anchor_ctr_offset
        self.base_anchors = self.gen_base_anchors()


    @property
    def num_base_anchors(self):
        return self.base_anchors.size(0)

    def gen_base_anchors(self):
        w = self.base_size
        h = self.base_size
        if self.ctr is None:
            x_ctr = 0.5 * (w - 1) #center point ?
            y_ctr = 0.5 * (h - 1)
        else:
            x_ctr, y_ctr = self.ctr

        h_ratios = torch.sqrt(self.ratios)
        w_ratios = 1 / h_ratios
        if self.scale_major:
            ws = (w * w_ratios[:, None] * self.scales[None, :]).view(-1)
            hs = (h * h_ratios[:, None] * self.scales[None, :]).view(-1)
        else:
            ws = (w * self.scales[:, None] * w_ratios[None, :]).view(-1)
            hs = (h * self.scales[:, None] * h_ratios[None, :]).view(-1)

        # yapf: disable
        #top left ang bottom right
        base_anchors = torch.stack(
            [
                x_ctr - 0.5 * (ws - 1), y_ctr - 0.5 * (hs - 1),
                x_ctr + 0.5 * (ws - 1), y_ctr + 0.5 * (hs - 1)
            ],
            dim=-1).round()
        # yapf: enable

        if self.use_rotate_box:
            # generate rotate anchors
            num_of_base_anchors = len(base_anchors)
            rotation_angles = len(self.rotate_range) # the amount of rotate angles
            base_anchors = base_anchors.repeat(rotation_angles, 1)
            rotate_range = self.rotate_range.repeat(num_of_base_anchors, 1)

            transform_angles = []
            for i in range(rotation_angles):
                 # [num_of_base_anchors, rotation_angles] -> [-1,1]
                transform_angles.append(rotate_range[: , i:i+1])

            rotate_range = torch.cat(transform_angles)
            base_anchors = torch.cat([base_anchors, rotate_range], 1)

        return base_anchors

    def _meshgrid(self, x, y, row_major=True):
        xx = x.repeat(len(y))
        yy = y.view(-1, 1).repeat(1, len(x)).view(-1)
        if row_major:
            return xx, yy
        else:
            return yy, xx

    def grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)

        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride

        if self.use_anchor_ctr_offset:
            offset = torch.tensor(stride / 2., dtype=torch.long).to(device)
            shift_x += offset
            shift_y += offset

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)
        # print("shift_xx shape", shift_xx.shape)
        # print("shift_xx", shift_xx)
        # print("shift_yy shape", shift_yy.shape)
        # print("shift_yy", shift_yy)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # print("shifts shape: ", shifts.shape)
        # print("shifts : \n", shifts)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        # print("base_anchors shape :", base_anchors[None, :, :].shape)
        # print("base_anchors shape :", base_anchors[None, :, :])
        # print("shifts shape :", shifts[:, None, :].shape)
        # print("shifts shape :", shifts[:, None, :])
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        # print("all_anchors", all_anchors.shape)
        # print("all_anchors", all_anchors)
        all_anchors = all_anchors.view(-1, 4)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...
        # print(all_anchors.shape)
        # print(all_anchors)

        return all_anchors

    def rotate_grid_anchors(self, featmap_size, stride=16, device='cuda'):
        base_anchors = self.base_anchors.to(device)
        # print("base_anchors: ", base_anchors)
        # print("featmap_size: ", featmap_size)
        feat_h, feat_w = featmap_size
        shift_x = torch.arange(0, feat_w, device=device) * stride
        shift_y = torch.arange(0, feat_h, device=device) * stride

        if self.use_anchor_ctr_offset:
            offset = torch.tensor(stride / 2., dtype=torch.long).to(device)
            shift_x += offset
            shift_y += offset

        shift_xx, shift_yy = self._meshgrid(shift_x, shift_y)

        angles = torch.zeros_like(shift_xx)

        # print("shift_xx shape : ", shift_xx.shape)
        # print("shift_xx : ", shift_xx)
        # print("shift_yy shape : ", shift_yy.shape)
        # print("shift_yy : ", shift_yy)
        shifts = torch.stack([shift_xx, shift_yy, shift_xx, shift_yy, angles], dim=-1)
        shifts = shifts.type_as(base_anchors)
        # print("shifts shape: ", shifts.shape)
        # print("shifts : \n", shifts)
        # first feat_w elements correspond to the first row of shifts
        # add A anchors (1, A, 4) to K shifts (K, 1, 4) to get
        # shifted anchors (K, A, 4), reshape to (K*A, 4)

        # print("base_anchors shape :", base_anchors[None, :, :].shape)
        # print("base_anchors shape :", base_anchors[None, :, :])
        # print("shifts shape :", shifts[:, None, :].shape)
        # print("shifts shape :", shifts[:, None, :])
        #base_anchors -> top left ang bottom right
        all_anchors = base_anchors[None, :, :] + shifts[:, None, :]
        # print("all_anchors", all_anchors.shape)
        # print("all_anchors", all_anchors[0])
        all_anchors = all_anchors.view(-1, 5)
        # first A rows correspond to A anchors of (0, 0) in feature map,
        # then (0, 1), (0, 2), ...

        return all_anchors

    def valid_flags(self, featmap_size, valid_size, device='cuda'):
        feat_h, feat_w = featmap_size
        valid_h, valid_w = valid_size
        assert valid_h <= feat_h and valid_w <= feat_w
        valid_x = torch.zeros(feat_w, dtype=torch.uint8, device=device)
        valid_y = torch.zeros(feat_h, dtype=torch.uint8, device=device)
        valid_x[:valid_w] = 1
        valid_y[:valid_h] = 1
        valid_xx, valid_yy = self._meshgrid(valid_x, valid_y)
        valid = valid_xx & valid_yy
        valid = valid[:,
                      None].expand(valid.size(0),
                                   self.num_base_anchors).contiguous().view(-1)
        return valid
