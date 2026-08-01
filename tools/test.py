# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel

from mmedit.apis import single_gpu_test, set_random_seed
from mmedit.datasets import build_dataloader, build_dataset
from mmedit.models import build_model
from mmedit.utils import setup_multi_processes



def parse_args():
    parser = argparse.ArgumentParser(description='mmediting tester (single GPU, no torchrun)')
    parser.add_argument('config', help='test config file path')
    parser.add_argument('checkpoint', help='checkpoint file')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--out', help='output result pickle file')
    parser.add_argument(
        '--save-path',
        default=None,
        type=str,
        help='path to store images and if not given, will not save image')
    parser.add_argument('--tmpdir', help='(unused in single-gpu test)')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
        'in xxx=yyy format will be merged into config file. If the value to '
        'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
        'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
        'Note that the quotation marks are necessary and that no white space '
        'is allowed.')
    parser.add_argument(
        '--gpu-id',
        type=int,
        default=0,
        help='which GPU to use (default: 0)')
    args = parser.parse_args()
    return args



def main():
    args = parse_args()

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)

    # set multi-process settings
    setup_multi_processes(cfg)

    # set cudnn_benchmark
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

    # no pretrained during testing
    if hasattr(cfg, 'model') and isinstance(cfg.model, dict):
        cfg.model.pretrained = None

    # set random seeds
    if args.seed is not None:
        print('set random seed to', args.seed)
        set_random_seed(args.seed, deterministic=args.deterministic)

    # build the dataset & dataloader
    dataset = build_dataset(cfg.data.test)
    loader_cfg = {
        **{k: cfg.data[k] for k in ['workers_per_gpu'] if k in cfg.data},
        **dict(samples_per_gpu=1, drop_last=False, shuffle=False, dist=False),
        **cfg.data.get('test_dataloader', {})
    }
    data_loader = build_dataloader(dataset, **loader_cfg)

    # select GPU
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_id)
        device_ids = [args.gpu_id]
    else:
        device_ids = []

    # build the model and load checkpoint
    model = build_model(cfg.model, train_cfg=None, test_cfg=cfg.get('test_cfg', None))

    # load to CPU first, then wrap for single-GPU
    _ = mmcv.runner.load_checkpoint(model, args.checkpoint, map_location='cpu')

    if torch.cuda.is_available():
        model = MMDataParallel(model, device_ids=device_ids)
    else:
        # CPU fallback (slow)
        model = MMDataParallel(model, device_ids=[])
        

    # run test
    save_image = args.save_path is not None
    outputs = single_gpu_test(
        model,
        data_loader,
        save_path=args.save_path,
        save_image=save_image
    )

    # print metrics (if dataset supports evaluate)
    if outputs and isinstance(outputs, list):
        print('')
        try:
            stats = dataset.evaluate(outputs)
            for k, v in stats.items():
                print(f'Eval-{k}: {v}')
            # append stats for optional dump
            outputs.append(stats)
        except Exception as e:
            # Some datasets do not implement evaluate or outputs are raw preds
            print(f'[Info] Skip evaluation: {e}')

    # save result pickle
    if args.out:
        print(f'writing results to {args.out}')
        mmcv.dump(outputs, args.out)


if __name__ == '__main__':
    main()