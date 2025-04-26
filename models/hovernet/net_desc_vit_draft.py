"""
================================================================================
Program Name: Nuclei Segmentation and Classification with HoVerIT
Version: 1.0
Author: David Qu
Institution: Yangtze River Delta Guozhi(Shanghai) Intelligient Medical Technology Co., Ltd
Email: davidqu01921@outlook.com
Date: 2025-04-08
================================================================================

Description:
This network is improved based on original HoVerNet architecture for nuclear segmentation and classfication in histopathology images.
It includes modules for training, testing, and visualizing segmentation results.

Usage:
To run this program, you can use the following command:
    python run_train.py --gpu='0'

Dependencies:
- Python 3.9
- PyTorch 1.10
- NumPy
- OpenCV

License:
This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Contact:
For any questions, suggestions, or bug reports, please contact David Qu at davidqu01921@outlook.com.

================================================================================
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict
from timm.models.swin_transformer import SwinTransformer

from .net_utils import DenseBlock, Net, TFSamepaddingLayer, UpSample2x
from .utils import crop_op

class HoverSwinNet(Net):
    def __init__(self, input_ch=3, nr_types=None, freeze=False, mode='original'):
        super().__init__()
        self.mode = mode
        self.freeze = freeze
        self.nr_types = nr_types
        self.output_ch = 3 if nr_types is None else 4

        self.window_size = 7  # Swin window size
        self.patch_size = 4   # Patch size for Swin

        # Swin Transformer encoder
        self.swin = SwinTransformer(
            img_size=256,
            patch_size=self.patch_size,
            in_chans=input_ch,
            embed_dim=96,
            depths=[2, 2, 6, 2],
            num_heads=[3, 6, 12, 24],
            window_size=self.window_size,
            drop_path_rate=0.2,
        )

        # Optional padding
        self.pad = TFSamepaddingLayer(ksize=7, stride=1) if mode == 'fast' else nn.Identity()

        # Match decoder input channels
        self.conv_bot = nn.Conv2d(768, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            u3 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(1024, 256, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(256, [1, ksize], [128, 32], 8, split=4)),
                ("convf", nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False)),
            ]))

            u2 = nn.Sequential(OrderedDict([
                ("conva", nn.Conv2d(512, 128, ksize, stride=1, padding=0, bias=False)),
                ("dense", DenseBlock(128, [1, ksize], [128, 32], 4, split=4)),
                ("convf", nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False)),
            ]))

            u1 = nn.Sequential(OrderedDict([
                ("conva/pad", TFSamepaddingLayer(ksize=ksize, stride=1)),
                ("conva", nn.Conv2d(256, 64, ksize, stride=1, padding=0, bias=False)),
            ]))

            u0 = nn.Sequential(OrderedDict([
                ("bn", nn.BatchNorm2d(64, eps=1e-5)),
                ("relu", nn.ReLU(inplace=True)),
                ("conv", nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True)),
            ]))

            return nn.Sequential(OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)]))

        ksize = 5 if mode == 'original' else 3
        self.decoder = nn.ModuleDict(OrderedDict([
            ("np", create_decoder_branch(ksize=ksize, out_ch=2)),
            ("hv", create_decoder_branch(ksize=ksize, out_ch=2)),
            ("tp", create_decoder_branch(ksize=ksize, out_ch=nr_types)) if nr_types else None,
        ]))

        self.upsample2x = UpSample2x()
        self.weights_init()

        self.align_d2 = nn.Conv2d(768, 1024, kernel_size=1)
        self.align_d1 = nn.Conv2d(384, 512, kernel_size=1)
        self.align_d0 = nn.Conv2d(192, 256, kernel_size=1)
        self.conv_bot = nn.Conv2d(768, 1024, kernel_size=1)



    def forward(self, imgs):
        orig_h, orig_w = imgs.shape[2], imgs.shape[3]
        imgs = imgs / 255.0

        # Pad to make H, W divisible by window size
        pad_h = (self.window_size - orig_h % self.window_size) % self.window_size
        pad_w = (self.window_size - orig_w % self.window_size) % self.window_size
        imgs = F.pad(imgs, (0, pad_w, 0, pad_h), mode='reflect')

        # Resize to 256x256
        imgs = F.interpolate(imgs, size=(256, 256), mode='bilinear', align_corners=False)
        print("preswin:",imgs.shape)
        # Swin forward
        x = self.swin.patch_embed(imgs)
        
        if hasattr(self.swin, 'absolute_pos_embed') and self.swin.absolute_pos_embed is not None:
            x = x + self.swin.absolute_pos_embed
        elif hasattr(self.swin, 'pos_embed') and self.swin.pos_embed is not None:
            x = x + self.swin.pos_embed

        x = self.swin.pos_drop(x)
        print("afterpos_drop:",x.shape)
        d = []

        # Collect feature maps from each Swin stage
        for i, layer in enumerate(self.swin.layers):
            x = layer(x)
            # print("afterswin:",x.shape)
            B, L, C = x.shape
            H = W = int(math.sqrt(L))
            feat = x.permute(0, 2, 1).reshape(B, C, H, W)
            print("after reshape:",feat.shape)
            d.append(feat)
        print("afterswin:",x.shape)
        # Bottom feature
        print(f"d[-1] has shape: {d[-1].shape}")
        print(f"d[-2] has shape: {d[-2].shape}")
        print(f"d[-3] has shape: {d[-3].shape}")
        print(f"d[-4] has shape: {d[-4].shape}")
        d3 = d[-1]  # 768 channels
        d3 = self.conv_bot(d3)  # 768 -> 1024 channels
        d[-1] = d3

        # Align d[-2], d[-3], d[-4]
        d[-2] = self.align_d2(d[-2])  # 768 → 1024
        d[-3] = self.align_d1(d[-3])  # 384 → 512
        d[-4] = self.align_d0(d[-4])  # 192 → 256

        print(f"d[-1] changed shape: {d[-1].shape}")
        print(f"d[-2] changed shape: {d[-2].shape}")
        print(f"d[-3] changed shape: {d[-3].shape}")
        print(f"d[-4] changed shape: {d[-4].shape}")

        # Optional crop
        d[0] = crop_op(d[0], [92, 92] if self.mode == 'fast' else [184, 184])
        d[1] = crop_op(d[1], [36, 36] if self.mode == 'fast' else [72, 72])
        print(f"d[0] changed shape: {d[0].shape}")
        print(f"d[1] changed shape: {d[1].shape}")
        print(f"d[2] changed shape: {d[2].shape}")
        print(f"d[3] changed shape: {d[3].shape}")

        # Decode
        out_dict = OrderedDict()
        for name, branch in self.decoder.items():
            if branch is None:
                continue
            u3_up = self.upsample2x(d[-1])

            # 对齐 spatial 尺寸
            if u3_up.shape[2:] != d[-2].shape[2:]:
                u3_up = F.interpolate(u3_up, size=d[-2].shape[2:], mode='bilinear', align_corners=False)

            u3 = u3_up + d[-2]
            u3 = branch[0](u3)

            u2 = self.upsample2x(u3)
            if u2.shape[2:] != d[-3].shape[2:]:
                u2 = F.interpolate(u2, size=d[-3].shape[2:], mode='bilinear', align_corners=False)
            u2 = u2 + d[-3]
            u2 = branch[1](u2)

            u1 = self.upsample2x(u2)
            if u1.shape[2:] != d[-4].shape[2:]:
                u1 = F.interpolate(u1, size=d[-4].shape[2:], mode='bilinear', align_corners=False)
            u1 = u1 + d[-4]
            u1 = branch[2](u1)

            u0 = branch[3](u1)

            # Remove padding
            out_dict[name] = u0[:, :, :orig_h, :orig_w]

        return out_dict

def create_model(mode=None, **kwargs):
    if mode not in ['original', 'fast']:
        raise ValueError(f"Unknown Model Mode {mode}")
    return HoverSwinNet(mode=mode, **kwargs)
