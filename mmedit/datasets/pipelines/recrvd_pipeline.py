import numpy as np
import torch
from ..registry import PIPELINES


def _rescale_raw01(x, black_level=240, white_level=4095):
    """RAW 预处理：减黑电平 + 按(white-black)归一到[0,1]。"""
    x = x.astype(np.float32)
    denom = float(white_level - black_level)
    x = (x - float(black_level)) / max(denom, 1.0)
    return np.clip(x, 0.0, 1.0)


def _to_tensor(lq, gt):
    """[T, H, W, C] → [T, C, H, W]（连续内存）。"""
    lq = np.ascontiguousarray(lq.transpose(0, 3, 1, 2))
    gt = np.ascontiguousarray(gt.transpose(0, 3, 1, 2))
    return torch.from_numpy(lq), torch.from_numpy(gt)


def _random_crop_3d(lq, gt, crop_size, t_sel, rng):
    """随机时空裁剪：随机取连续 t_sel 帧 + 空间随机裁剪。输入 [T,H,W,C]。
       注意：T 取 noisy/clean 两端的最小值，避免一端切空。
    """
    Tl, Hl, Wl, _ = lq.shape
    Tg, Hg, Wg, _ = gt.shape
    T = min(Tl, Tg)
    th, tw = crop_size

    if t_sel is None or t_sel > T:
        t_sel = T
    t0 = rng.integers(0, T - t_sel + 1)

    if th > Hl or th > Hg or tw > Wl or tw > Wg:
        raise ValueError(f'crop_size {crop_size} exceeds data size '
                         f'noisy={(Hl,Wl)}, clean={(Hg,Wg)}')

    Hc, Wc = min(Hl, Hg), min(Wl, Wg)
    y0 = rng.integers(0, Hc - th + 1)
    x0 = rng.integers(0, Wc - tw + 1)

    lq_c = lq[t0:t0 + t_sel, y0:y0 + th, x0:x0 + tw, :].copy()
    gt_c = gt[t0:t0 + t_sel, y0:y0 + th, x0:x0 + tw, :].copy()
    return lq_c, gt_c, int(t0)


def _rand_flip_hw(lq, gt, rng, p_h=0.5, p_w=0.5):
    """仅空间翻转（不翻时间轴）。为避免负 stride，操作后立刻 copy。"""
    if rng.random() < p_w:
        lq = lq[:, :, ::-1, :].copy()
        gt = gt[:, :, ::-1, :].copy()
    if rng.random() < p_h:
        lq = lq[:, ::-1, :, :].copy()
        gt = gt[:, ::-1, :, :].copy()
    return lq, gt


def _rand_transpose_hw(lq, gt, rng, p=0.5):
    """随机 H↔W 置换。"""
    if rng.random() < p:
        lq = np.transpose(lq, (0, 2, 1, 3)).copy()
        gt = np.transpose(gt, (0, 2, 1, 3)).copy()
    return lq, gt


@PIPELINES.register_module()
class ReCRVDTrainPipeline:
    """训练：随机时空裁剪 + 空间增强 + RAW归一化 + 转tensor。
    假设 dataset 已做 Bayer→4通道打包：lq/gt 形状 [T,H,W,4]（uint16 或 float）。
    """
    def __init__(self,
                 crop_size=(128, 128),
                 flip_ratio=None,            # 兼容旧参数 (t,h,w)，不使用 t
                 transpose_ratio=None,       # 兼容旧参数
                 flip_prob_h=None,
                 flip_prob_w=None,
                 transpose_prob=None,
                 black_level=240,
                 white_level=4095,
                 seed=None,
                 **kwargs):
        self.crop_size = tuple(crop_size)
        self.black_level = int(black_level)
        self.white_level = int(white_level)
        self.rng = np.random.default_rng(seed)

        # 兼容老参 flip_ratio=(t,h,w)，时间翻转不做
        if flip_prob_h is None or flip_prob_w is None:
            if flip_ratio is not None:
                flip_prob_h = flip_prob_h if flip_prob_h is not None else float(flip_ratio[1])
                flip_prob_w = flip_prob_w if flip_prob_w is not None else float(flip_ratio[2])
            else:
                flip_prob_h = 0.5 if flip_prob_h is None else float(flip_prob_h)
                flip_prob_w = 0.5 if flip_prob_w is None else float(flip_prob_w)
        self.flip_prob_h = float(flip_prob_h)
        self.flip_prob_w = float(flip_prob_w)

        if transpose_prob is None and transpose_ratio is not None:
            transpose_prob = transpose_ratio
        self.transpose_prob = 0.5 if transpose_prob is None else float(transpose_prob)

    def __call__(self, results):
        # 归一化到 [0,1]
        lq = _rescale_raw01(results['lq'], self.black_level, self.white_level)
        gt = _rescale_raw01(results['gt'], self.black_level, self.white_level)

        # 随机时空裁剪
        t_sel = results.get('num_input_frames', None)
        lq, gt, start = _random_crop_3d(lq, gt, self.crop_size, t_sel, self.rng)

        # 空间随机增强（不翻时间轴）
        lq, gt = _rand_flip_hw(lq, gt, self.rng, self.flip_prob_h, self.flip_prob_w)
        lq, gt = _rand_transpose_hw(lq, gt, self.rng, self.transpose_prob)

        # 转 tensor（连续内存）
        lq_t, gt_t = _to_tensor(lq, gt)

        return dict(
            lq=lq_t,
            gt=gt_t,
            lq_path=results.get('lq_path', ''),
            gt_path=results.get('gt_path', ''),
            key=results.get('key', ''),
            iso=results.get('iso', ''),
            start_frame=start,
        )


@PIPELINES.register_module()
class ReCRVDTestPipeline:
    """验证/测试：整段整幅，无随机增强；RAW→[0,1] 后转 tensor。"""
    def __init__(self, black_level=240, white_level=4095, **kwargs):
        super().__init__()
        self.black_level = int(black_level)
        self.white_level = int(white_level)

    def __call__(self, results):
        lq = _rescale_raw01(results['lq'], self.black_level, self.white_level)
        gt = _rescale_raw01(results['gt'], self.black_level, self.white_level)
        lq_t, gt_t = _to_tensor(lq, gt)
        return dict(
            lq=lq_t,
            gt=gt_t,
            lq_path=results.get('lq_path', ''),
            gt_path=results.get('gt_path', ''),
            key=results.get('key', ''),
            iso=results.get('iso', ''),
            start_frame=0,
        )


@PIPELINES.register_module()
class ReCRVDValPipeline(ReCRVDTestPipeline):
    """与 Test 完全一致（为了兼容 cfg 中的名字）。"""
    pass
