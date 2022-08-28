_base_ = [
    '../../_base_/datasets/dotav1.py', '../../_base_/default_runtime.py'
]

img_scale = (1920, 1080)
samples_per_gpu = 4

angle_version = 'le90'
dataset_type = 'Mydata'

model = dict(
    type='ByteTrack',
    detector=dict(
    type='OrientedRCNN',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch',
        init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5),
    rpn_head=dict(
        type='OrientedRPNHead',
        in_channels=256,
        feat_channels=256,
        version=angle_version,
        anchor_generator=dict(
            type='AnchorGenerator',
            scales=[8],
            ratios=[0.5, 1.0, 2.0],
            strides=[4, 8, 16, 32, 64]),
        bbox_coder=dict(
            type='MidpointOffsetCoder',
            angle_range=angle_version,
            target_means=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            target_stds=[1.0, 1.0, 1.0, 1.0, 0.5, 0.5]),
        loss_cls=dict(
            type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
        loss_bbox=dict(
            type='SmoothL1Loss', beta=0.1111111111111111, loss_weight=1.0)),
    roi_head=dict(
        type='OrientedStandardRoIHead',
        bbox_roi_extractor=dict(
            type='RotatedSingleRoIExtractor',
            roi_layer=dict(
                type='RoIAlignRotated',
                out_size=7,
                sample_num=2,
                clockwise=True),
            out_channels=256,
            featmap_strides=[4, 8, 16, 32]),
        bbox_head=dict(
            type='RotatedShared2FCBBoxHead',
            in_channels=256,
            fc_out_channels=1024,
            roi_feat_size=7,
            num_classes=2,
            bbox_coder=dict(
                type='DeltaXYWHAOBBoxCoder',
                angle_range=angle_version,
                norm_factor=None,
                edge_swap=True,
                proj_xy=True,
                target_means=(.0, .0, .0, .0, .0),
                target_stds=(0.1, 0.1, 0.2, 0.2, 0.1)),
            reg_class_agnostic=True,
            loss_cls=dict(
                type='CrossEntropyLoss', use_sigmoid=False, loss_weight=1.0),
            loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))),
    init_cfg=dict(
            type='Pretrained',
            checkpoint=  # noqa: E251
            'G:/大创项目/mmTracking/mmtrack/mmtrack_test/epoch_40_mul.pth'  # noqa: E501
        ),
    train_cfg=dict(
        rpn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.7,
                neg_iou_thr=0.3,
                min_pos_iou=0.3,
                match_low_quality=True,
                ignore_iof_thr=-1),
            sampler=dict(
                type='RandomSampler',
                num=256,
                pos_fraction=0.5,
                neg_pos_ub=-1,
                add_gt_as_proposals=False),
            allowed_border=0,
            pos_weight=-1,
            debug=False),
        rpn_proposal=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            assigner=dict(
                type='MaxIoUAssigner',
                pos_iou_thr=0.5,
                neg_iou_thr=0.5,
                min_pos_iou=0.5,
                match_low_quality=False,
                iou_calculator=dict(type='RBboxOverlaps2D'),
                ignore_iof_thr=-1),
            sampler=dict(
                type='RRandomSampler',
                num=512,
                pos_fraction=0.25,
                neg_pos_ub=-1,
                add_gt_as_proposals=True),
            pos_weight=-1,
            debug=False)),
    test_cfg=dict(
        rpn=dict(
            nms_pre=2000,
            max_per_img=2000,
            nms=dict(type='nms', iou_threshold=0.8),
            min_bbox_size=0),
        rcnn=dict(
            nms_pre=2000,
            min_bbox_size=0,
            score_thr=0.05,
            nms=dict(iou_thr=0.1),
            max_per_img=2000))),
    motion=dict(type='KalmanFilter'),
    tracker=dict(
        type='ByteTracker',
        obj_score_thrs=dict(high=0.6, low=0.1),
        init_track_thr=0.7,
        weight_iou_with_det_scores=True,
        match_iou_thrs=dict(high=0.1, low=0.5, tentative=0.3),
        num_frames_retain=30))

# train_pipeline = [
#     dict(
#         type='Mosaic',
#         img_scale=img_scale,
#         pad_val=114.0,
#         bbox_clip_border=False),
#     dict(
#         type='RandomAffine',
#         scaling_ratio_range=(0.1, 2),
#         border=(-img_scale[0] // 2, -img_scale[1] // 2),
#         bbox_clip_border=False),
#     dict(
#         type='MixUp',
#         img_scale=img_scale,
#         ratio_range=(0.8, 1.6),
#         pad_val=114.0,
#         bbox_clip_border=False),
#     dict(type='YOLOXHSVRandomAug'),
#     dict(type='RandomFlip', flip_ratio=0.5),
#     dict(
#         type='Resize',
#         img_scale=img_scale,
#         keep_ratio=True,
#         bbox_clip_border=False),
#     dict(type='Pad', size_divisor=32, pad_val=dict(img=(114.0, 114.0, 114.0))),
#     dict(type='FilterAnnotations', min_gt_bbox_wh=(1, 1), keep_empty=False),
#     dict(type='DefaultFormatBundle'),
#     dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels'])
# ]
#
# test_pipeline = [
#     dict(type='LoadImageFromFile'),
#     dict(
#         type='MultiScaleFlipAug',
#         img_scale=img_scale,
#         flip=False,
#         transforms=[
#             dict(type='Resize', keep_ratio=True),
#             dict(type='RandomFlip'),
#             dict(
#                 type='Normalize',
#                 mean=[0.0, 0.0, 0.0],
#                 std=[1.0, 1.0, 1.0],
#                 to_rgb=False),
#             dict(
#                 type='Pad',
#                 size_divisor=32,
#                 pad_val=dict(img=(114.0, 114.0, 114.0))),
#             dict(type='ImageToTensor', keys=['img']),
#             dict(type='VideoCollect', keys=['img'])
#         ])
# ]
# data = dict(
#     samples_per_gpu=samples_per_gpu,
#     workers_per_gpu=4,
#     persistent_workers=True,
#     train=dict(
#         _delete_=True,
#         type='MultiImageMixDataset',
#         dataset=dict(
#             type='CocoDataset',
#             ann_file=[
#                 'data/MOT17/annotations/half-train_cocoformat.json',
#                 'data/crowdhuman/annotations/crowdhuman_train.json',
#                 'data/crowdhuman/annotations/crowdhuman_val.json'
#             ],
#             img_prefix=[
#                 'data/MOT17/train', 'data/crowdhuman/train',
#                 'data/crowdhuman/val'
#             ],
#             classes=('pedestrian', ),
#             pipeline=[
#                 dict(type='LoadImageFromFile'),
#                 dict(type='LoadAnnotations', with_bbox=True)
#             ],
#             filter_empty_gt=False),
#         pipeline=train_pipeline),
#     val=dict(
#         pipeline=test_pipeline,
#         interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)),
#     test=dict(
#         pipeline=test_pipeline,
#         interpolate_tracks_cfg=dict(min_num_frames=5, max_num_frames=20)))
#
# # optimizer
# # default 8 gpu
# optimizer = dict(
#     type='SGD',
#     lr=0.001 / 8 * samples_per_gpu,
#     momentum=0.9,
#     weight_decay=5e-4,
#     nesterov=True,
#     paramwise_cfg=dict(norm_decay_mult=0.0, bias_decay_mult=0.0))
# optimizer_config = dict(grad_clip=None)
#
# # some hyper parameters
# total_epochs = 80
# num_last_epochs = 10
# resume_from = None
# interval = 5
#
# # learning policy
# lr_config = dict(
#     policy='YOLOX',
#     warmup='exp',
#     by_epoch=False,
#     warmup_by_epoch=True,
#     warmup_ratio=1,
#     warmup_iters=1,
#     num_last_epochs=num_last_epochs,
#     min_lr_ratio=0.05)
#
# custom_hooks = [
#     dict(
#         type='YOLOXModeSwitchHook',
#         num_last_epochs=num_last_epochs,
#         priority=48),
#     dict(
#         type='SyncNormHook',
#         num_last_epochs=num_last_epochs,
#         interval=interval,
#         priority=48),
#     dict(
#         type='ExpMomentumEMAHook',
#         resume_from=resume_from,
#         momentum=0.0001,
#         priority=49)
# ]
#
# checkpoint_config = dict(interval=1)
# evaluation = dict(metric=['bbox', 'track'], interval=1)
# search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']
#
# # you need to set mode='dynamic' if you are using pytorch<=1.5.0
# fp16 = dict(loss_scale=dict(init_scale=512.))
# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    warmup_iters=100,
    warmup_ratio=1.0 / 100,
    step=[3])
# runtime settings
total_epochs = 4
evaluation = dict(metric=['bbox', 'track'], interval=1)
search_metrics = ['MOTA', 'IDF1', 'FN', 'FP', 'IDs', 'MT', 'ML']