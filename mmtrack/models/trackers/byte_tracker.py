# Copyright (c) OpenMMLab. All rights reserved.
import lap
import numpy as np
import torch
from mmcv.runner import force_fp32
from mmdet.core import bbox_overlaps

from mmtrack.core.bbox import bbox_cxcyah_to_xyxy, bbox_xyxy_to_cxcyah
from mmtrack.models import TRACKERS
from .base_tracker import BaseTracker


@TRACKERS.register_module()
class ByteTracker(BaseTracker):
    """Tracker for ByteTrack.

    Args:
        obj_score_thrs (dict): Detection score threshold for matching objects.
            - high (float): Threshold of the first matching. Defaults to 0.6.
            - low (float): Threshold of the second matching. Defaults to 0.1.
        init_track_thr (float): Detection score threshold for initializing a
            new tracklet. Defaults to 0.7.
        weight_iou_with_det_scores (bool): Whether using detection scores to
            weight IOU which is used for matching. Defaults to True.
        match_iou_thrs (dict): IOU distance threshold for matching between two
            frames.
            - high (float): Threshold of the first matching. Defaults to 0.1.
            - low (float): Threshold of the second matching. Defaults to 0.5.
            - tentative (float): Threshold of the matching for tentative
                tracklets. Defaults to 0.3.
        num_tentatives (int, optional): Number of continuous frames to confirm
            a track. Defaults to 3.
        init_cfg (dict or list[dict], optional): Initialization config dict.
            Defaults to None.
    """

    def __init__(self,
                 obj_score_thrs=dict(high=0.6, low=0.1),
                 init_track_thr=0.7,
                 weight_iou_with_det_scores=True,
                 match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
                 num_tentatives=3,
                 init_cfg=None,
                 **kwargs):
        super().__init__(init_cfg=init_cfg, **kwargs)
        self.obj_score_thrs = obj_score_thrs
        self.init_track_thr = init_track_thr

        self.weight_iou_with_det_scores = weight_iou_with_det_scores
        self.match_iou_thrs = match_iou_thrs

        self.num_tentatives = num_tentatives

    @property
    def confirmed_ids(self):
        """Confirmed ids in the tracker."""
        # print("666666!!!!!!")
        # print(self.tracks.items())
        # for track in self.tracks.items():
        #     print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        #     print("track.tentative")
        #     print(track)
        #     print(track[1]['tentative'])
        ids = [id for id, track in self.tracks.items() if not track[1]['tentative']]
        # print("PPP")
        # print(ids)
        return ids

    @property
    def unconfirmed_ids(self):
        """Unconfirmed ids in the tracker."""
        ids = [id for id, track in self.tracks.items() if track[1]['tentative']]
        return ids

    def init_track(self, id, obj):
        """Initialize a track."""
        super().init_track(id, obj)
        # print("(((((((((((((((((((((((((((((()))))))))))))))))))))))))))))")
        if self.tracks[id].frame_ids[-1] == 0:
            # print("LLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLLL")
            self.tracks[id].tentative = False
        else:
            # print("IIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIIII")
            self.tracks[id].tentative = True
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.initiate(
            bbox)

    def update_track(self, id, obj):
        """Update a track."""
        super().update_track(id, obj)
        # print("(((((((((((((((((((((((((((((()))))))))))))))))))))))))))))")
        # print("self.tracks[id].tentative&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&")
        # print(self.tracks[id].tentative)
        # print("FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF")
        if self.tracks[id].tentative:
            if len(self.tracks[id]['bboxes']) >= self.num_tentatives:
                # print("GGGGGGGGGGGGGGGGGGGGGGGGGGGGGG")
                self.tracks[id].tentative = False
        bbox = bbox_xyxy_to_cxcyah(self.tracks[id].bboxes[-1])  # size = (1, 4)
        assert bbox.ndim == 2 and bbox.shape[0] == 1
        bbox = bbox.squeeze(0).cpu().numpy()
        track_label = self.tracks[id]['labels'][-1]
        label_idx = self.memo_items.index('labels')
        obj_label = obj[label_idx]
        assert obj_label == track_label
        self.tracks[id].mean, self.tracks[id].covariance = self.kf.update(
            self.tracks[id].mean, self.tracks[id].covariance, bbox)

    def pop_invalid_tracks(self, frame_id):
        """Pop out invalid tracks."""
        invalid_ids = []
        for k, v in self.tracks.items():
            # case1: disappeared frames >= self.num_frames_retrain
            case1 = frame_id - v['frame_ids'][-1] >= self.num_frames_retain
            # case2: tentative tracks but not matched in this frame
            case2 = v.tentative and v['frame_ids'][-1] != frame_id
            if case1 or case2:
                invalid_ids.append(k)
        for invalid_id in invalid_ids:
            self.tracks.pop(invalid_id)

    def assign_ids(self,
                   ids,
                   det_bboxes,
                   det_labels,
                   weight_iou_with_det_scores=False,
                   match_iou_thr=0.5):
        """Assign ids.

        Args:
            ids (list[int]): Tracking ids.
            det_bboxes (Tensor): of shape (N, 5)
            weight_iou_with_det_scores (bool, optional): Whether using
                detection scores to weight IOU which is used for matching.
                Defaults to False.
            match_iou_thr (float, optional): Matching threshold.
                Defaults to 0.5.

        Returns:
            tuple(int): The assigning ids.
        """
        # get track_bboxes
        # print("ERERERERERRERE")
        # print(det_labels)
        track_bboxes = np.zeros((0, 4))
        # print(track_bboxes)
        # print("121214343434")
        # print(ids)
        for id in ids:
            # print(self.tracks[id])
            track_bboxes = np.concatenate(
                (track_bboxes, self.tracks[id].mean[:4][None]), axis=0)
        # print("444444")
        # for id in ids:
        #     print(self.tracks[id].mean[:4][None])
        #     print(self.tracks[id].mean[:4][None].shape)
        # print(track_bboxes.shape)
        # print(track_bboxes)
        track_bboxes = torch.from_numpy(track_bboxes).to(det_bboxes)
        # print("6666666")
        # print(track_bboxes.shape)
        # print(track_bboxes[:,0])
        # print(track_bboxes)
        track_bboxes = bbox_cxcyah_to_xyxy(track_bboxes)
        # print(track_bboxes)
        # print(det_bboxes)
        # print(det_bboxes[:, 0])
        ##在这里要把检测框格式转换成xyxy
        det_box = np.array(det_bboxes.cpu())
        # print(det_box)
        # print(det_box.shape)
        det_boxessss1 = det_box[:, 0]-0.5*det_box[:, 2]
        det_boxessss2 = det_box[:, 1]-0.5*det_box[:, 3]
        det_boxessss3 = det_box[:, 0]+0.5*det_box[:, 2]
        det_boxessss4 = det_box[:, 1]+0.5*det_box[:, 3]
        # print("%^%^%^%^%^%^%")
        # print(len(det_boxessss1))
        # print(det_boxessss1)
        # print(det_boxessss2)
        # print(det_boxessss3)
        # print(det_boxessss4)
        # print(det_boxessss1.shape)
        if len(det_boxessss1)!= 0:
            det_boxessss_test = np.array([[det_boxessss1[0], det_boxessss2[0], det_boxessss3[0], det_boxessss4[0]]])
            # print(det_boxessss_test.shape)
            for i in range(1, len(det_boxessss1)):
                db = np.array([[det_boxessss1[i], det_boxessss2[i], det_boxessss3[i], det_boxessss4[i]]])
                # print(db)
                det_boxessss_test = np.r_[det_boxessss_test, db]
            # print("YFFYFYFYYFFY")
            # print(det_boxessss_test)
        # det_boxessss = det_boxessss1[0:]
        # det_boxessss = np.append(det_boxessss1, det_boxessss2)
        # det_boxessss = np.append(det_boxessss, det_boxessss3)
        # det_boxessss = np.append(det_boxessss, det_boxessss4)
        # print(det_boxessss.shape)
        # det_boxessss = det_boxessss1 + det_boxessss2 + det_boxessss3 + det_boxessss4
        # print("dfdfdfdf")
        # print(track_bboxes.shape)
        # print(det_boxessss)
        # det_boxessss = np.array([det_boxessss])
        # print("oioioioioi")
        # print(det_boxessss.size)
        # print(det_boxessss.shape)
        if len(det_boxessss1)!= 0:
            det_boxessss = torch.from_numpy(det_boxessss_test).to(det_bboxes)
        else:
            det_boxessss = np.zeros((0, 4))
            det_boxessss = torch.from_numpy(det_boxessss).to(det_bboxes)

        # print(det_boxessss)
        # print(track_bboxes)
        # print(det_bboxes[:, :4])
        # print(det_boxessss.size(-1))
        # print(det_boxessss.size(0))

        # print("opopop[opopopo")
        # print(det_boxessss)
        # print(track_bboxes)


        # if det_boxessss.size(-1) == 0:
        #     print("bnbnbnbnnbnb")
        #     det_boxessss = np.zeros((0, 4))
        #     det_boxessss = torch.from_numpy(det_boxessss).to(det_bboxes)
        #     print(det_boxessss)
        #     print(det_boxessss.shape)

        # det_box = np.array([[det_bboxes[:, 0], det_bboxes[:, 1], det_bboxes[:, 2], det_bboxes[:, 3]]])
        # print(det_box)

        ##修改
        # compute distance
        ious = bbox_overlaps(track_bboxes, det_boxessss)
        # ious = bbox_overlaps(track_bboxes, det_bboxes[:, :4])
        if weight_iou_with_det_scores:
            ious *= det_bboxes[:, 5][None]
        # print("ious****************************")
        # print(ious)
        # support multi-class association
        # print("det_bboxes.device")
        # print(det_bboxes.device)
        # print("OIOIOIOIOI")
        # print(len(ids))
        # for id in ids:
            # print(self.tracks[id]['labels'][-1])
        track_labels = torch.tensor([
            self.tracks[id]['labels'][-1] for id in ids
        ]).to(det_bboxes.device)
        det_labels = det_labels.to(device=torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'))
        # print(det_labels)
        # print(track_labels)
        cate_match = det_labels[None, :] == track_labels[:, None]
        # print("QWQWQWQWQWQWQWQ")
        # print(cate_match)
        # to avoid det and track of different categories are matched
        cate_cost = (1 - cate_match.int()) * 1e6

        dists = (1 - ious + cate_cost).cpu().numpy()
        # print("FGFGFGFGFGF")
        # print(dists)
        # print(dists.size)

        # bipartite match
        if dists.size > 0:
            cost, row, col = lap.lapjv(
                dists, extend_cost=True, cost_limit=1 - match_iou_thr)
            # print("RRRRRRRRRRRRRRRRRRRRRRR")
            # print(row)
            # print(col)
        else:
            row = np.zeros(len(ids)).astype(np.int32) - 1
            col = np.zeros(len(det_bboxes)).astype(np.int32) - 1
        return row, col

    @force_fp32(apply_to=('img', 'bboxes'))
    def track(self,
              img,
              img_metas,
              model,
              bboxes,
              labels,
              frame_id,
              rescale=False,
              **kwargs):
        """Tracking forward function.

        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
            model (nn.Module): MOT model.
            bboxes (Tensor): of shape (N, 5).
            labels (Tensor): of shape (N, ).
            frame_id (int): The id of current frame, 0-index.
            rescale (bool, optional): If True, the bounding boxes should be
                rescaled to fit the original scale of the image. Defaults to
                False.
        Returns:
            tuple: Tracking results.
        """
        if not hasattr(self, 'kf'):
            self.kf = model.motion


        # print("2323232323322")
        # print(self.empty)
        # print(bboxes.size(0))
        if self.empty or bboxes.size(0) == 0:
            valid_inds = bboxes[:, -1] > self.init_track_thr
            bboxes = bboxes[valid_inds]
            labels = labels[valid_inds]
            num_new_tracks = bboxes.size(0)
            ids = torch.arange(self.num_tracks,
                               self.num_tracks + num_new_tracks).to(labels)
            self.num_tracks += num_new_tracks

        else:
            # 0. init
            ids = torch.full((bboxes.size(0), ),
                             -1,
                             dtype=labels.dtype,
                             device=labels.device)
            # print("ids")
            # print(ids)
            # get the detection bboxes for the first association
            first_det_inds = bboxes[:, -1] > self.obj_score_thrs['high']
            first_det_bboxes = bboxes[first_det_inds]
            first_det_labels = labels[first_det_inds]
            first_det_ids = ids[first_det_inds]

            # get the detection bboxes for the second association
            second_det_inds = (~first_det_inds) & (
                bboxes[:, -1] > self.obj_score_thrs['low'])
            second_det_bboxes = bboxes[second_det_inds]
            second_det_labels = labels[second_det_inds]
            second_det_ids = ids[second_det_inds]

            # 1. use Kalman Filter to predict current location
            # print("************************(*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)")
            # print(self.confirmed_ids)
            # print("************************(*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)*)")
            for id in self.confirmed_ids:
                # track is lost in previous frame
                if self.tracks[id].frame_ids[-1] != frame_id - 1:
                    self.tracks[id].mean[7] = 0
                (self.tracks[id].mean,
                 self.tracks[id].covariance) = self.kf.predict(
                     self.tracks[id].mean, self.tracks[id].covariance)

            # 2. first match
            first_match_track_inds, first_match_det_inds = self.assign_ids(
                self.confirmed_ids, first_det_bboxes, first_det_labels,
                self.weight_iou_with_det_scores, self.match_iou_thrs['high'])
            # print("89989898")
            # print(first_match_track_inds)
            # print(first_match_det_inds)
            # '-1' mean a detection box is not matched with tracklets in
            # previous frame
            valid = first_match_det_inds > -1
            # valid = first_match_det_inds == -1
            # print(valid)
            first_det_ids[valid] = torch.tensor(
                self.confirmed_ids)[first_match_det_inds[valid]].to(labels)

            first_match_det_bboxes = first_det_bboxes[valid]
            first_match_det_labels = first_det_labels[valid]
            first_match_det_ids = first_det_ids[valid]
            assert (first_match_det_ids > -1).all()

            first_unmatch_det_bboxes = first_det_bboxes[~valid]
            first_unmatch_det_labels = first_det_labels[~valid]
            first_unmatch_det_ids = first_det_ids[~valid]
            assert (first_unmatch_det_ids == -1).all()

            # 3. use unmatched detection bboxes from the first match to match
            # the unconfirmed tracks
            (tentative_match_track_inds,
             tentative_match_det_inds) = self.assign_ids(
                 self.unconfirmed_ids, first_unmatch_det_bboxes,
                 first_unmatch_det_labels, self.weight_iou_with_det_scores,
                 self.match_iou_thrs['tentative'])
            valid = tentative_match_det_inds > -1
            first_unmatch_det_ids[valid] = torch.tensor(self.unconfirmed_ids)[
                tentative_match_det_inds[valid]].to(labels)

            # 4. second match for unmatched tracks from the first match
            first_unmatch_track_ids = []
            for i, id in enumerate(self.confirmed_ids):
                # tracklet is not matched in the first match
                case_1 = first_match_track_inds[i] == -1
                # tracklet is not lost in the previous frame
                case_2 = self.tracks[id].frame_ids[-1] == frame_id - 1
                if case_1 and case_2:
                    first_unmatch_track_ids.append(id)

            second_match_track_inds, second_match_det_inds = self.assign_ids(
                first_unmatch_track_ids, second_det_bboxes, second_det_labels,
                False, self.match_iou_thrs['low'])
            valid = second_match_det_inds > -1
            second_det_ids[valid] = torch.tensor(first_unmatch_track_ids)[
                second_match_det_inds[valid]].to(ids)

            # 5. gather all matched detection bboxes from step 2-4
            # we only keep matched detection bboxes in second match, which
            # means the id != -1
            valid = second_det_ids > -1
            bboxes = torch.cat(
                (first_match_det_bboxes, first_unmatch_det_bboxes), dim=0)
            bboxes = torch.cat((bboxes, second_det_bboxes[valid]), dim=0)

            # print("090909099")
            # print(first_match_det_ids)
            # print(first_unmatch_det_ids)
            labels = torch.cat(
                (first_match_det_labels, first_unmatch_det_labels), dim=0)
            labels = torch.cat((labels, second_det_labels[valid]), dim=0)

            ids = torch.cat((first_match_det_ids, first_unmatch_det_ids),
                            dim=0)
            ids = torch.cat((ids, second_det_ids[valid]), dim=0)

            # 6. assign new ids
            # print("121212121212")
            # print(ids)
            # print(ids == -1)
            new_track_inds = ids == -1
            ids[new_track_inds] = torch.arange(
                self.num_tracks,
                self.num_tracks + new_track_inds.sum()).to(labels)
            self.num_tracks += new_track_inds.sum()
            # print(ids)

        self.update(ids=ids, bboxes=bboxes, labels=labels, frame_ids=frame_id)
        print("VVVVVVVVVVVVVVVVVVVVVVV")
        print(ids)
        return bboxes, labels, ids
