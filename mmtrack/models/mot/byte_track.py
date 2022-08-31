# Copyright (c) OpenMMLab. All rights reserved.

import torch
from mmdet.models import build_detector

from mmtrack.core import outs2results, results2outs
from ..builder import MODELS, build_motion, build_tracker
from .base import BaseMultiObjectTracker


@MODELS.register_module()
class ByteTrack(BaseMultiObjectTracker):
    """ByteTrack: Multi-Object Tracking by Associating Every Detection Box.

    This multi object tracker is the implementation of `ByteTrack
    <https://arxiv.org/abs/2110.06864>`_.

    Args:
        detector (dict): Configuration of detector. Defaults to None.
        tracker (dict): Configuration of tracker. Defaults to None.
        motion (dict): Configuration of motion. Defaults to None.
        init_cfg (dict): Configuration of initialization. Defaults to None.
    """

    def __init__(self,
                 detector=None,
                 tracker=None,
                 motion=None,
                 init_cfg=None):
        super().__init__(init_cfg)

        if detector is not None:
            self.detector = build_detector(detector)

        if motion is not None:
            self.motion = build_motion(motion)

        if tracker is not None:
            self.tracker = build_tracker(tracker)

    def forward_train(self, *args, **kwargs):
        """Forward function during training."""
        return self.detector.forward_train(*args, **kwargs)

    def simple_test(self, img, img_metas, rescale=False, public_bboxes = None, **kwargs):
        """Test without augmentations.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            rescale (bool, optional): If False, then returned bboxes and masks
                will fit the scale of img, otherwise, returned bboxes and masks
                will fit the scale of original image shape. Defaults to False.

        Returns:
            dict[str : list(ndarray)]: The tracking results.
        """
        # frame_id = img_metas[0].get('frame_id', -1)
        # if frame_id == 0:
        #     self.tracker.reset()
        #
        # # print("self.detector:")
        # # print(self.detector)
        # det_results = self.detector.simple_test(
        #     img, img_metas, rescale=rescale)
        # assert len(det_results) == 1, 'Batch inference is not supported.'
        # bbox_results = det_results[0]
        # print("bbox_results")
        # print(bbox_results)
        # num_classes = len(bbox_results)
        # print("num_classes")
        # print(num_classes)
        # outs_det = results2outs(bbox_results=bbox_results)
        # det_bboxes = torch.from_numpy(outs_det['bboxes']).to(img)
        # det_labels = torch.from_numpy(outs_det['labels']).to(img).long()
        # print("下面是检测结果：")
        # print(det_bboxes)
        # print("下面是bbox数量")
        # print(len(det_bboxes))
        # print(det_labels)
        # print("下面是label数量：")
        # print(len(det_labels))
        # print("img:")
        # print(img)
        # print(img_metas)
        frame_id = img_metas[0].get('frame_id', -1)
        if frame_id == 0:
            self.tracker.reset()

        x = self.detector.extract_feat(img)
        if hasattr(self.detector, 'roi_head'):
            # TODO: check whether this is the case
            if public_bboxes is not None:
                public_bboxes = [_[0] for _ in public_bboxes]
                proposals = public_bboxes
            else:
                proposals = self.detector.rpn_head.simple_test_rpn(
                    x, img_metas)

            # print("important")
            # print(x)
            # print(img_metas)
            # print(proposals)
            # print(self.detector.roi_head.test_cfg)
            det_bboxes, det_labels = self.detector.roi_head.simple_test_bboxes(
                x,
                img_metas,
                proposals,
                self.detector.roi_head.test_cfg,
                rescale=rescale)
            # TODO: support batch inference
            # print(det_bboxes)
            # print(det_labels)
            # print("zhongyao")
            det_bboxes = det_bboxes[0]
            det_labels = det_labels[0]
            print("下面是重要信息")
            print(det_bboxes)
            print(det_labels)
            num_classes = self.detector.roi_head.bbox_head.num_classes
        elif hasattr(self.detector, 'bbox_head'):
            outs = self.detector.bbox_head(x)
            result_list = self.detector.bbox_head.get_bboxes(
                *outs, img_metas=img_metas, rescale=rescale)
            # TODO: support batch inference
            det_bboxes = result_list[0][0]
            det_labels = result_list[0][1]
            num_classes = self.detector.bbox_head.num_classes
        else:
            raise TypeError('detector must has roi_head or bbox_head.')

        track_bboxes, track_labels, track_ids = self.tracker.track(
            img=img,
            img_metas=img_metas,
            model=self,
            bboxes=det_bboxes,
            labels=det_labels,
            frame_id=frame_id,
            rescale=rescale,
            **kwargs)
        # print("+++++++++++++++++++++++++++++++++++++")
        # print(track_ids)
        track_results = outs2results(
            bboxes=track_bboxes,
            labels=track_labels,
            ids=track_ids,
            num_classes=num_classes)
        det_results = outs2results(
            bboxes=det_bboxes, labels=det_labels, num_classes=num_classes)

        return dict(
            det_bboxes=det_results['bbox_results'],
            track_bboxes=track_results['bbox_results'])
