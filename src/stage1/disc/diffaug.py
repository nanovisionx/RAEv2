# this file is taken from https://github.com/autonomousvision/stylegan-t/blob/36ab80ce76237fefe03e65e9b3161c040ae888e3/training/diffaug.py
import math

import torch
import torch.nn.functional as F


class DiffAug(object):
    def __init__(self, prob=1.0, cutout=0.2): # todo: swin ratio = 0.5, T&XL = 0.2
        self.grids = {}
        self.prob = abs(prob)
        self.using_cutout = prob > 0
        self.cutout = cutout
        self.img_channels = -1
        self.last_blur_radius = -1
        self.last_blur_kernel_h = self.last_blur_kernel_w = None

    def __str__(self):
        return f'DiffAug(p={self.prob}, cutout={self.cutout if self.using_cutout else 0})'
    def get_grids(self, B, x, y, dev):
        if (B, x, y) in self.grids:
            return self.grids[(B, x, y)]

        self.grids[(B, x, y)] = ret = torch.meshgrid(
            torch.arange(B, dtype=torch.long, device=dev),
            torch.arange(x, dtype=torch.long, device=dev),
            torch.arange(y, dtype=torch.long, device=dev),
            indexing='ij'
        )
        return ret

    def aug(self, BCHW: torch.Tensor, warmup_blur_schedule: float = 0) -> torch.Tensor:
        # warmup blurring
        if BCHW.dtype != torch.float32:
            BCHW = BCHW.float()
        if warmup_blur_schedule > 0:
            self.img_channels = BCHW.shape[1]
            sigma0 = (BCHW.shape[-2] * 0.5) ** 0.5
            sigma = sigma0 * warmup_blur_schedule
            blur_radius = math.floor(sigma * 3)           # 3-sigma is enough for Gaussian
            print(f'Warmup blurring with sigma={sigma}, radius={blur_radius}')
            if blur_radius >= 1:
                if self.last_blur_radius != blur_radius:
                    self.last_blur_radius = blur_radius
                    gaussian = torch.arange(-blur_radius, blur_radius + 1, dtype=torch.float32, device=BCHW.device)
                    gaussian = gaussian.mul_(1/sigma).square_().neg_().exp2_()
                    gaussian.div_(gaussian.sum())     # normalize
                    self.last_blur_kernel_h = gaussian.view(1, 1, 2*blur_radius+1, 1).repeat(self.img_channels, 1, 1, 1).contiguous()
                    self.last_blur_kernel_w = gaussian.view(1, 1, 1, 2*blur_radius+1).repeat(self.img_channels, 1, 1, 1).contiguous()

                BCHW = F.pad(BCHW, [blur_radius, blur_radius, blur_radius, blur_radius], mode='reflect')
                BCHW = F.conv2d(input=BCHW, weight=self.last_blur_kernel_h, bias=None, groups=self.img_channels)
                BCHW = F.conv2d(input=BCHW, weight=self.last_blur_kernel_w, bias=None, groups=self.img_channels)
                # BCHW = filter2d(BCHW, f.div_(f.sum()))  # no need to specify padding (filter2d will add padding in itself based on filter size)

        if self.prob < 1e-6:
            return BCHW
        trans, color, cut = torch.rand(3) <= self.prob
        trans, color, cut = trans.item(), color.item(), cut.item()
        B, dev = BCHW.shape[0], BCHW.device
        rand01 = torch.rand(7, B, 1, 1, device=dev) if (trans or color or cut) else None

        raw_h, raw_w = BCHW.shape[-2:]
        if trans:
            ratio = 0.125
            delta_h = round(raw_h * ratio)
            delta_w = round(raw_w * ratio)
            translation_h = rand01[0].mul(delta_h+delta_h+1).floor().long() - delta_h
            translation_w = rand01[1].mul(delta_w+delta_w+1).floor().long() - delta_w
            # translation_h = torch.randint(-delta_h, delta_h+1, size=(B, 1, 1), device=dev)
            # translation_w = torch.randint(-delta_w, delta_w+1, size=(B, 1, 1), device=dev)

            grid_B, grid_h, grid_w = self.get_grids(B, raw_h, raw_w, dev)
            grid_h = (grid_h + translation_h).add_(1).clamp_(0, raw_h+1)
            grid_w = (grid_w + translation_w).add_(1).clamp_(0, raw_w+1)
            bchw_pad = F.pad(BCHW, [1, 1, 1, 1, 0, 0, 0, 0])
            BCHW = bchw_pad.permute(0, 2, 3, 1).contiguous()[grid_B, grid_h, grid_w].permute(0, 3, 1, 2).contiguous()

        if color:
            BCHW = BCHW.add(rand01[2].unsqueeze(-1).sub(0.5))
            # BCHW.add_(torch.rand(B, 1, 1, 1, dtype=BCHW.dtype, device=dev).sub_(0.5))
            bchw_mean = BCHW.mean(dim=1, keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[3].unsqueeze(-1).mul(2)).add_(bchw_mean)
            # BCHW.sub_(bchw_mean).mul_(torch.rand(B, 1, 1, 1, dtype=BCHW.dtype, device=dev).mul_(2)).add_(bchw_mean)
            bchw_mean = BCHW.mean(dim=(1, 2, 3), keepdim=True)
            BCHW = BCHW.sub(bchw_mean).mul(rand01[4].unsqueeze(-1).add(0.5)).add_(bchw_mean)
            # BCHW.sub_(bchw_mean).mul_(torch.rand(B, 1, 1, 1, dtype=BCHW.dtype, device=dev).add_(0.5)).add_(bchw_mean)

        if self.using_cutout and cut:
            ratio = self.cutout # todo: styleswin ratio = 0.5, T&XL = 0.2
            cutout_h = round(raw_h * ratio)
            cutout_w = round(raw_w * ratio)
            offset_h = rand01[5].mul(raw_h + (1 - cutout_h % 2)).floor().long()
            offset_w = rand01[6].mul(raw_w + (1 - cutout_w % 2)).floor().long()
            # offset_h = torch.randint(0, raw_h + (1 - cutout_h % 2), size=(B, 1, 1), device=dev)
            # offset_w = torch.randint(0, raw_w + (1 - cutout_w % 2), size=(B, 1, 1), device=dev)

            grid_B, grid_h, grid_w = self.get_grids(B, cutout_h, cutout_w, dev)
            grid_h = (grid_h + offset_h).sub_(cutout_h // 2).clamp(min=0, max=raw_h - 1)
            grid_w = (grid_w + offset_w).sub_(cutout_w // 2).clamp(min=0, max=raw_w - 1)
            mask = torch.ones(B, raw_h, raw_w, dtype=BCHW.dtype, device=dev)
            mask[grid_B, grid_h, grid_w] = 0
            BCHW = BCHW.mul(mask.unsqueeze(1))

        return BCHW
