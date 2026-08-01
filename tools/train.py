import argparse
import copy
import os
import os.path as osp
import time

import mmcv
import torch
import torch.distributed as dist
from mmcv import Config, DictAction
from mmcv.runner import init_dist

from mmedit import __version__
from mmedit.apis import init_random_seed, set_random_seed, train_model
from mmedit.datasets import build_dataset
from mmedit.models import build_model
from mmedit.utils import collect_env, get_root_logger, setup_multi_processes


def parse_args():
    parser = argparse.ArgumentParser(description='Train an editor')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', help='the dir to save logs and models')
    parser.add_argument('--resume-from', help='the checkpoint file to resume from')
    parser.add_argument('--no-validate', action='store_true',
                        help='whether not to evaluate the checkpoint during training')
    parser.add_argument('--gpus', type=int, default=1,
                        help='number of gpus to use (only applicable to non-distributed training)')
    parser.add_argument('--seed', type=int, default=None, help='random seed')
    parser.add_argument('--diff_seed', action='store_true',
                        help='Whether or not set different seeds for different ranks')
    parser.add_argument('--deterministic', action='store_true',
                        help='whether to set deterministic options for CUDNN backend.')
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='override some settings in the used config, the key-value pair in xxx=yyy format will be merged into config file. If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" Note that the quotation marks are necessary and that no white space is allowed.')
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'], default='none', help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true', help='automatically scale lr with the number of gpus')
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


def main():
  
    args = parse_args()
    cfg = Config.fromfile(args.config)

    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)


    setup_multi_processes(cfg)

   
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True

  
    if args.work_dir is not None:
        cfg.work_dir = args.work_dir
    if args.resume_from is not None:
        cfg.resume_from = args.resume_from
    cfg.gpus = args.gpus

    if args.autoscale_lr:
       
        if hasattr(cfg, 'optimizers') and 'generator' in cfg.optimizers:
            cfg.optimizers.generator.lr = cfg.optimizers.generator.lr * cfg.gpus / 8
        elif hasattr(cfg, 'optimizer'):
            cfg.optimizer['lr'] = cfg.optimizer['lr'] * cfg.gpus / 8

    distributed = (args.launcher != 'none')
    if distributed:
        init_dist(args.launcher, **getattr(cfg, 'dist_params', dict(backend='nccl')))

  
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(log_file=log_file, log_level=cfg.log_level)

  
    env_info_dict = collect_env.collect_env()
    env_info = '\n'.join([f'{k}: {v}' for k, v in env_info_dict.items()])
    dash_line = '-' * 60 + '\n'
    logger.info('Environment info:\n' + dash_line + env_info + '\n' + dash_line)

    logger.info(f'Distributed training: {distributed}')
    logger.info(f'mmedit Version: {__version__}')
    logger.info('Config:\n' + cfg.text)

  
    seed = init_random_seed(args.seed)
    if args.diff_seed and dist.is_available() and dist.is_initialized():
        seed = seed + dist.get_rank()
    logger.info(f'Set random seed to {seed}, deterministic: {args.deterministic}')
    set_random_seed(seed, deterministic=args.deterministic)
    cfg.seed = seed


    model = build_model(cfg.model, train_cfg=cfg.train_cfg, test_cfg=cfg.test_cfg)
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        val_dataset.pipeline = cfg.data.train.pipeline
        datasets.append(build_dataset(val_dataset))


    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmedit_version=__version__,
            config=cfg.text,
        )

    meta = dict()
    if cfg.get('exp_name', None) is None:
        cfg['exp_name'] = osp.splitext(osp.basename(cfg.work_dir))[0]
    meta['exp_name'] = cfg.exp_name
    meta['mmedit Version'] = __version__
    meta['seed'] = seed
    meta['env_info'] = env_info

 
    train_model(
        model,
        datasets,
        cfg,
        distributed=distributed,
        validate=(not args.no_validate),
        timestamp=timestamp,
        meta=meta
    )


if __name__ == '__main__':
    main()

