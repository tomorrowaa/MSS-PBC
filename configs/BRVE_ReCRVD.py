exp_name = 'BRVE_ReCRVD'


ReCRVD_ROOT = './datasets/ReCRVD'
DATA_LIST_ROOT = './data_list'

custom_imports = dict(
    imports=[
        'mmedit.models.backbones.bnnvd',
        'mmedit.datasets.recrvd_dataset',
        'mmedit.datasets.pipelines.recrvd_pipeline'
    ],
    allow_failed_imports=False
)

# ------------------ model ------------------
model = dict(
    type='BRVE',
    clip_length=50,
    generator=dict(
        type='BNNVD1',
        mid_channels=24,
        feat_extract_blocks=3,
        num_unets=1,
        unet_n_feat=[24, 48, 96],
        unet_n_block=[1, 3, 3],
        stage1_n_feat=[24, 48, 48, 48],
        task='Raw2Raw'
    ),
    pixel_loss=dict(
        type='GuidedFrequencyLoss',
        epsilon=1e-3,
        omega0=0.50, omegaF=0.80,
        num_epochs=100,
        num_stages=5,
        mode='static',
        loss_threshold=0.05,
        reduction='mean'
    )
)

# ------------------ train/test cfg ------------------
train_cfg = dict(fix_iter=-1)
test_cfg = dict(
    metrics=['PSNR', 'SSIM'],
    gt_format='raw',
    used_isp='ReCRVD_isp',
    tile=256,
    tile_overlap=32,
    video_resolution=(1080 // 2, 1920 // 2),
    frame_freq=10
)

custom_hooks = [
    dict(
        type='GFLIterSchedulerHook',
        total_iters=100000,
        omega0=0.50, omegaF=0.80,
        attr_paths=[
            "module.pixel_loss", "pixel_loss",
            "module.loss_gfl", "loss_gfl",
            "module.generator.pixel_loss", "generator.pixel_loss",
            "module.backbone.loss_gfl", "backbone.loss_gfl",
        ],
        log=True
    ),
]

# ------------------ pipelines ------------------
train_pipeline = [
    dict(
        type='ReCRVDTrainPipeline',
        crop_size=(128, 128),
        black_level=240,
        white_level=4095
    ),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key', 'iso', 'start_frame']
    ),
]

val_pipeline = [
    dict(type='ReCRVDTestPipeline', black_level=240, white_level=4095),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key', 'iso']
    ),
]

test_pipeline = [
    dict(type='ReCRVDTestPipeline', black_level=240, white_level=4095),
    dict(
        type='Collect',
        keys=['lq', 'gt'],
        meta_keys=['lq_path', 'gt_path', 'key', 'iso']
    ),
]

# ------------------ data ------------------
data = dict(
    workers_per_gpu=0,
    train_dataloader=dict(samples_per_gpu=1, drop_last=True, persistent_workers=False),
    val_dataloader  =dict(samples_per_gpu=1, persistent_workers=False),
    test_dataloader =dict(samples_per_gpu=1, persistent_workers=False, workers_per_gpu=1),

    train=dict(
        type='ReCRVDDataset',
        lq_folder=f'{ReCRVD_ROOT}/wb_scene_noisy',
        gt_folder=f'{ReCRVD_ROOT}/wb_scene_clean_postprocessed',
        pipeline=train_pipeline,
        ann_file=f'{DATA_LIST_ROOT}/ReCRVD_train_scenes.txt',
        num_input_frames=10,
        pack_order='GBRG',
        memorize=False,  
        test_mode=False
    ),

    val=dict(
        type='ReCRVDDataset',
        lq_folder=f'{ReCRVD_ROOT}/wb_scene_noisy',
        gt_folder=f'{ReCRVD_ROOT}/wb_scene_clean_postprocessed',
        pipeline=val_pipeline,
        ann_file=f'{DATA_LIST_ROOT}/ReCRVD_val_scenes.txt',
        pack_order='GBRG',
        memorize=True,
        test_mode=True
    ),

    test=dict(
        type='ReCRVDDataset',
        lq_folder=f'{ReCRVD_ROOT}/wb_scene_noisy',
        gt_folder=f'{ReCRVD_ROOT}/wb_scene_clean_postprocessed',
        pipeline=test_pipeline,
        ann_file=f'{DATA_LIST_ROOT}/ReCRVD_test_scenes.txt',
        pack_order='GBRG',
        memorize=True,
        test_mode=True
    ),
)

# ------------------ optimizer / sched ------------------
optimizers = dict(generator=dict(type='Adam', lr=2e-4, betas=(0.9, 0.99)))

total_iters = 100000
lr_config = dict(
    policy='CosineRestart',
    by_epoch=False,
    periods=[100000],
    restart_weights=[1],
    min_lr=1e-7
)

checkpoint_config = dict(interval=2000, save_optimizer=True, by_epoch=False)
evaluation = dict(interval=2000, save_image=False)
log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='TensorboardLoggerHook'),
    ]
)
visual_config = None

# ------------------ runtime ------------------
log_level = 'INFO'
work_dir = f'./experiments/{exp_name}'
load_from = None
resume_from = None
workflow = [('train', 1)]
find_unused_parameters = True
