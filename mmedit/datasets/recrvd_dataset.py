# Copyright (c) OpenMMLab. All rights reserved.
import glob
import os
import os.path as osp
import re
import numpy as np
import copy
from collections import defaultdict

import mmcv
import cv2
from mmcv.runner import get_dist_info

from .base_dataset import BaseDataset
from .registry import DATASETS


def _natural_key(path):
    """用于自然排序（适配 'wb_noisy_10_0.tiff' 等多数字文件名）。"""
    name = osp.splitext(osp.basename(path))[0]
    return [int(s) if s.isdigit() else s for s in re.split(r'(\d+)', name)]


def _pack_bayer_4ch(raw, cfa='RGGB'):
    """单通道 Bayer RAW 打成 4 通道 (R,G1,G2,B)。支持 {'RGGB','GRBG','GBRG','BGGR'}。"""
    H, W = raw.shape
    H2, W2 = H - H % 2, W - W % 2
    raw = raw[:H2, :W2]

    if cfa == 'RGGB':
        r  = raw[0::2, 0::2]; g1 = raw[0::2, 1::2]; g2 = raw[1::2, 0::2]; b  = raw[1::2, 1::2]
    elif cfa == 'GRBG':
        g1 = raw[0::2, 0::2]; r  = raw[0::2, 1::2]; b  = raw[1::2, 0::2]; g2 = raw[1::2, 1::2]
    elif cfa == 'GBRG':
        g1 = raw[0::2, 0::2]; b  = raw[0::2, 1::2]; r  = raw[1::2, 0::2]; g2 = raw[1::2, 1::2]
    elif cfa == 'BGGR':
        b  = raw[0::2, 0::2]; g1 = raw[0::2, 1::2]; g2 = raw[1::2, 0::2]; r  = raw[1::2, 1::2]
    else:
        raise ValueError(f'Unknown CFA pattern: {cfa}')

    return np.stack([r, g1, g2, b], axis=2)


def _load_clean_clip(gt_dir, cfa='GBRG'):
    """读取 clean 目录下 25 帧 TIFF 并打包。"""
    img_paths = glob.glob(osp.join(gt_dir, '*.tiff')) + glob.glob(osp.join(gt_dir, '*.tif'))
    if len(img_paths) == 0:
        raise FileNotFoundError(f'No TIFF files found in {gt_dir}')
    img_paths.sort(key=_natural_key)
    frames = []
    for p in img_paths:
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2:
            raise IOError(f'Bad RAW in {p}')
        frames.append(_pack_bayer_4ch(raw, cfa=cfa).astype(np.uint16))
    return np.stack(frames, axis=0)  # [T, H/2, W/2, 4]


def _load_noisy_per_frame_random(lq_iso_dir, cfa='GBRG', rng=None):
    """从 noisy ISO 目录逐帧随机选择 repeat，组 25 帧序列并打包。"""
    paths = glob.glob(osp.join(lq_iso_dir, '*.tiff')) + glob.glob(osp.join(lq_iso_dir, '*.tif'))
    if len(paths) == 0:
        raise FileNotFoundError(f'No TIFF files found in {lq_iso_dir}')

    # 解析帧号/重复号：wb_noisy_<frame>_<rep>.tif(f)
    pat = re.compile(r'.*_(\d+)_(\d+)\.tif{1,2}f?$', re.IGNORECASE)
    by_frame = {}  # frame_idx -> {rep_idx: path}
    for p in paths:
        m = pat.match(osp.basename(p))
        if not m:
            continue
        fi, ri = int(m.group(1)), int(m.group(2))
        by_frame.setdefault(fi, {})[ri] = p

    frame_ids = sorted(by_frame.keys())
    if len(frame_ids) == 0:
        raise RuntimeError(f'Cannot parse frame/repeat under {lq_iso_dir}')
    if rng is None:
        rng = np.random.default_rng()

    frames = []
    for fi in frame_ids:  # 通常是 1..25
        rep_dict = by_frame[fi]
        rep_i = int(rng.choice(sorted(rep_dict.keys())))  # 逐帧随机一个可用 repeat
        p = rep_dict[rep_i]
        raw = cv2.imread(p, cv2.IMREAD_UNCHANGED)
        if raw is None or raw.ndim != 2:
            raise IOError(f'Bad RAW in {p}')
        frames.append(_pack_bayer_4ch(raw, cfa=cfa).astype(np.uint16))
    return np.stack(frames, axis=0)  # [T, H/2, W/2, 4]


@DATASETS.register_module()
class ReCRVDDataset(BaseDataset):
    """ReCRVD TIFF 数据集（RAW→RAW）。

    目录结构：
      noisy: /.../ReCRVD/wb_scene_noisy/<scene>/isoXXXX/*.tiff
      clean: /.../ReCRVD/wb_scene_clean_postprocessed/<scene>/*.tiff

    Args:
        lq_folder (str): noisy 根目录
        gt_folder (str): clean 根目录
        pipeline (list): pipeline
        ann_file (str): 每行一个 scene 名
        memorize (bool): 是否缓存 clip（训练逐帧随机建议 False）
        pack_order (str): CFA，'RGGB'/'GRBG'/'GBRG'/'BGGR'
        num_input_frames (None|int): 训练每次取的帧数；None=整段
        test_mode (bool): 测试模式
    """

    def __init__(self,
                 lq_folder,
                 gt_folder,
                 pipeline,
                 ann_file,
                 memorize=False,
                 pack_order='GBRG',
                 num_input_frames=None,
                 test_mode=True):
        super().__init__(pipeline, test_mode)
        self.lq_folder = str(lq_folder)
        self.gt_folder = str(gt_folder)
        self.ann_file = ann_file
        self.memorize = bool(memorize)
        self.pack_order = pack_order

        if num_input_frames is not None and num_input_frames <= 0:
            raise ValueError('"num_input_frames" must be None or positive.')
        self.num_input_frames = num_input_frames

        print('ReCRVDDataset: pack_order =', pack_order)

        self.lq_cache = dict()
        self.gt_cache = dict()
        self.data_infos = self.load_annotations()

        # RNG 用于逐帧随机
        self.rng = np.random.default_rng()

    def load_annotations(self):
        """构建样本索引：为每个 (scene, iso) noisy clip 配对 clean clip。"""
        data_infos = []
        scenes = mmcv.list_from_file(self.ann_file)

        for scene in scenes:
            scene = scene.strip()
            if not scene:
                continue

            lq_scene_path = osp.join(self.lq_folder, scene)
            if not osp.isdir(lq_scene_path):
                raise FileNotFoundError(f'Missing noisy scene dir: {lq_scene_path}')

            # noisy 端按 ISO 子目录（无 iso* 也兼容）
            iso_dirs = sorted(glob.glob(osp.join(lq_scene_path, 'iso*')))
            iso_dirs = [d for d in iso_dirs if osp.isdir(d)]
            if len(iso_dirs) == 0:
                iso_dirs = [lq_scene_path]  # isoNA

            gt_path = osp.join(self.gt_folder, scene)
            if not osp.isdir(gt_path):
                raise FileNotFoundError(f'Missing gt dir: {gt_path}')

            for lq_iso_path in iso_dirs:
                iso_name = osp.basename(lq_iso_path) if lq_iso_path != lq_scene_path else 'isoNA'
                data_infos.append(dict(
                    lq=None, gt=None,
                    lq_path=lq_iso_path,
                    gt_path=gt_path,
                    key=f'{scene}/{iso_name}',
                    scene=scene,
                    iso=iso_name,
                    num_input_frames=self.num_input_frames,
                ))
        return data_infos

    def _load_lq_noisy(self, iso_dir):
        """noisy：逐帧随机 repeat（每次 __getitem__ 都会重新随机）。"""
        # 训练要逐帧随机 => 不建议缓存；若强行缓存会固定第一次随机的结果
        return _load_noisy_per_frame_random(iso_dir, cfa=self.pack_order, rng=self.rng)

    def _load_gt_clean(self, gt_dir):
        return _load_clean_clip(gt_dir, cfa=self.pack_order)

    def prepare_train_data(self, idx):
        results = copy.deepcopy(self.data_infos[idx])
        lqp, gtp = results['lq_path'], results['gt_path']

        if self.memorize:
            # clean 可缓存
            if gtp not in self.gt_cache:
                rank, world_size = get_dist_info()
                print(f'{rank}/{world_size} loading [GT] {gtp}')
                self.gt_cache[gtp] = self._load_gt_clean(gtp)
            gt = self.gt_cache[gtp]
            # noisy 逐帧随机：不缓存（否则固定第一次随机）
            lq = self._load_lq_noisy(lqp)
        else:
            gt = self._load_gt_clean(gtp)
            lq = self._load_lq_noisy(lqp)

        results['lq'] = lq  # [T, H/2, W/2, 4], uint16
        results['gt'] = gt
        results['sequence_length'] = int(min(lq.shape[0], gt.shape[0]))  # 通常 25
        return self.pipeline(results)

    def prepare_test_data(self, idx):
        # 测试/验证：如果你也想逐帧随机，可和训练一致；若想可复现，改为固定 repeat=0 的实现
        results = copy.deepcopy(self.data_infos[idx])
        lqp, gtp = results['lq_path'], results['gt_path']

        # 这里保留与训练相同的逐帧随机；如需固定 repeat=0，可自行替换为专门的加载函数
        gt = self._load_gt_clean(gtp)
        lq = self._load_lq_noisy(lqp)

        results['lq'] = lq
        results['gt'] = gt
        results['sequence_length'] = int(min(lq.shape[0], gt.shape[0]))
        return self.pipeline(results)

    def __getitem__(self, idx):
        return self.prepare_test_data(idx) if self.test_mode else self.prepare_train_data(idx)

    def clear_cache(self):
        self.lq_cache.clear()
        self.gt_cache.clear()

    def evaluate(self, results, logger=None):
        """按 ISO 分桶 + 总体均值。"""
        if not isinstance(results, list):
            raise TypeError(f'results must be a list, got {type(results)}')
        assert len(results) == len(self), 'Length mismatch between results and dataset.'

        grouped = defaultdict(lambda: defaultdict(list))
        overall = defaultdict(list)

        for res in results:
            iso_key = res.get('iso', None)
            for metric, val in res['eval_result'].items():
                overall[metric].append(val)
                if iso_key is not None:
                    grouped[iso_key][metric].append(val)

        out = {}
        for k, m2v in grouped.items():
            for metric, vals in m2v.items():
                out[f'{metric}@{k}'] = float(np.mean(vals))
        for metric, vals in overall.items():
            out[metric] = float(np.mean(vals))
        return out
