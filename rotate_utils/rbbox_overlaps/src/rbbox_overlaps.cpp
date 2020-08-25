// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#include <torch/extension.h>

#define CHECK_CUDA(x) AT_CHECK(x.type().is_cuda(), #x, " must be a CUDAtensor ")

at::Tensor rbbox_overlaps(const at::Tensor boxes, const at::Tensor query_boxes, const int gpu_device_id);

at::Tensor rbbox_iou(const at::Tensor& dets_1, const at::Tensor& dets_2, const int& device_id) {
  CHECK_CUDA(dets_1);
  if (dets_1.numel() == 0)
    return at::empty({0}, dets_1.options().dtype(at::kLong).device(at::kCPU));
  return rbbox_overlaps(dets_1, dets_2, device_id);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("rbbox_iou", &rbbox_iou, "rbbox_iou");
}
