# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict
import torch
from torch.nn import functional as F

from detectron2.structures.boxes import Boxes, BoxMode

from ..structures import (
    DensePoseChartPredictorOutput,
    DensePoseChartResult,
    DensePoseChartResultWithConfidences,
)
from . import resample_fine_and_coarse_segm_to_bbox
from .base import IntTupleBox, make_int_box


def resample_uv_tensors_to_bbox(
    u: torch.Tensor,
    v: torch.Tensor,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
    im_size: tuple,
) -> torch.Tensor: # changed line
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        u (tensor [1, C, H, W] of float): U coordinates
        v (tensor [1, C, H, W] of float): V coordinates
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)
    u_bbox = F.interpolate(u, (h, w), mode="bilinear", align_corners=False)
    v_bbox = F.interpolate(v, (h, w), mode="bilinear", align_corners=False)
    canvas_u = torch.zeros(1, 25, im_size[1], im_size[0], device=u.device) # new line
    canvas_v = torch.zeros(1, 25, im_size[1], im_size[0], device=v.device) # new line
    canvas_u[:, :, y : y + h, x : x + w] = u_bbox # new line
    canvas_v[:, :, y : y + h, x : x + w] = v_bbox # new line
    uv = torch.zeros([2, im_size[1], im_size[0]], dtype=torch.float32, device=u.device) # changed line
    for part_id in range(1, canvas_u.size(1)): # changed line
        uv[0][labels == part_id] = canvas_u[0, part_id][labels == part_id] # changed line
        uv[1][labels == part_id] = canvas_v[0, part_id][labels == part_id] # changed line
    return uv

def resample_uv_to_bbox(
    predictor_output: DensePoseChartPredictorOutput,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
    im_size: tuple,
) -> torch.Tensor:
    """
    Resamples U and V coordinate estimates for the given bounding box

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be resampled
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled U and V coordinates - a tensor [2, H, W] of float
    """
    return resample_uv_tensors_to_bbox(
        predictor_output.u,
        predictor_output.v,
        labels,
        box_xywh_abs,
        im_size,
    )


def densepose_chart_predictor_output_to_result(
    predictor_output: DensePoseChartPredictorOutput, boxes: Boxes
) -> DensePoseChartResult:
    """
    Convert densepose chart predictor outputs to results

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be converted to results, must contain only 1 output
        boxes (Boxes): bounding box that corresponds to the predictor output,
            must contain only 1 bounding box
    Return:
       DensePose chart-based result (DensePoseChartResult)
    """
    assert len(predictor_output) == 1 and len(boxes) == 1, (
        f"Predictor output to result conversion can operate only single outputs"
        f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
    )

    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(predictor_output, box_xywh).squeeze(0)
    uv = resample_uv_to_bbox(predictor_output, labels, box_xywh)
    return DensePoseChartResult(labels=labels, uv=uv)


def resample_confidences_to_bbox(
    predictor_output: DensePoseChartPredictorOutput,
    labels: torch.Tensor,
    box_xywh_abs: IntTupleBox,
) -> Dict[str, torch.Tensor]:
    """
    Resamples confidences for the given bounding box

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output to be resampled
        labels (tensor [H, W] of long): labels obtained by resampling segmentation
            outputs for the given bounding box
        box_xywh_abs (tuple of 4 int): bounding box that corresponds to predictor outputs
    Return:
       Resampled confidences - a dict of [H, W] tensors of float
    """

    x, y, w, h = box_xywh_abs
    w = max(int(w), 1)
    h = max(int(h), 1)

    confidence_names = [
        "sigma_1",
        "sigma_2",
        "kappa_u",
        "kappa_v",
        "fine_segm_confidence",
        "coarse_segm_confidence",
    ]
    confidence_results = {key: None for key in confidence_names}
    confidence_names = [
        key for key in confidence_names if getattr(predictor_output, key) is not None
    ]
    confidence_base = torch.zeros([h, w], dtype=torch.float32, device=predictor_output.u.device)

    # assign data from channels that correspond to the labels
    for key in confidence_names:
        resampled_confidence = F.interpolate(
            getattr(predictor_output, key), (h, w), mode="bilinear", align_corners=False
        )
        result = confidence_base.clone()
        for part_id in range(1, predictor_output.u.size(1)):
            if resampled_confidence.size(1) != predictor_output.u.size(1):
                # confidence is not part-based, don't try to fill it part by part
                continue
            result[labels == part_id] = resampled_confidence[0, part_id][labels == part_id]

        if resampled_confidence.size(1) != predictor_output.u.size(1):
            # confidence is not part-based, fill the data with the first channel
            # (targeted for segmentation confidences that have only 1 channel)
            result = resampled_confidence[0, 0]

        confidence_results[key] = result

    return confidence_results  # pyre-ignore[7]


def densepose_chart_predictor_output_to_result_with_confidences(
    predictor_output: DensePoseChartPredictorOutput, boxes: Boxes, im_size: tuple
) -> DensePoseChartResultWithConfidences: # changed line
    """
    Convert densepose chart predictor outputs to results

    Args:
        predictor_output (DensePoseChartPredictorOutput): DensePose predictor
            output with confidences to be converted to results, must contain only 1 output
        boxes (Boxes): bounding box that corresponds to the predictor output,
            must contain only 1 bounding box
    Return:
       DensePose chart-based result with confidences (DensePoseChartResultWithConfidences)
    """
    assert len(predictor_output) == 1 and len(boxes) == 1, (
        f"Predictor output to result conversion can operate only single outputs"
        f", got {len(predictor_output)} predictor outputs and {len(boxes)} boxes"
    )

    boxes_xyxy_abs = boxes.tensor.clone()
    boxes_xywh_abs = BoxMode.convert(boxes_xyxy_abs, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    box_xywh = make_int_box(boxes_xywh_abs[0])

    labels = resample_fine_and_coarse_segm_to_bbox(
        predictor_output, box_xywh, im_size
    ).squeeze(0) #changed line
    uv = resample_uv_to_bbox(predictor_output, labels, box_xywh, im_size) # changed line
    confidences = resample_confidences_to_bbox(predictor_output, labels, box_xywh)
    return DensePoseChartResultWithConfidences(labels=labels, uv=uv, **confidences)
