from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys

import mmcv
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch import Tensor as t, Tensor
import torch.nn.functional as F

from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack
from mmseg.core import mean_iou


class RandomAttack(DecisionBlackBoxAttack):
    """
    SignFlip
    """

    def __init__(self, epsilon, p, resize_factor, max_queries, lb, ub, batch_size):
        super().__init__(max_queries=max_queries,
                         epsilon=epsilon,
                         p=p,
                         lb=lb,
                         ub=ub,
                         batch_size=batch_size)
        self.resize_factor = resize_factor

    def _config(self):
        return {
            "p": self.p,
            "epsilon": self.epsilon,
            "lb": self.lb,
            "ub": self.ub,
            "attack_name": self.__class__.__name__
        }

    def _perturb(self, x, y,img,xs,mkdir):

        c, h, w = x['img'][0].shape[1:]
        xt_a = t(np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], 3, h, w])).repeat(1, 1, 1, 1)
        init_noise=xt_a
        xt_a = xt_a + x['img'][0]

        temp = x['img'][0].clone()
        temp = dict(img=temp)
        temp['img_metas'] = x['img_metas']
        temp['img'] = temp['img'].to("cuda")
        temp['img'] = [temp['img']]

        tt = xt_a
        tt = dict(img=tt)
        tt['img_metas'] = x['img_metas']
        tt['img'] = tt['img'].to("cuda")
        tt['img'] = [tt['img']]
        out = self.model(return_loss=False, **temp)[0]
        out_iter = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
        miou_begin = np.nanmean(mean_iou([out_iter.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])
        min_miou=miou_begin
        print("random")
        for iter in range(200):
            noise=torch.rand_like(x['img'][0])*2-1
            noise=noise.clamp(min=-self.epsilon/16,max=self.epsilon/16)
            noise=noise+tt['img'][0]-temp['img'][0]
            noise=noise.clamp(min=-self.epsilon,max=self.epsilon)

            tt = (temp['img'][0]+noise).clamp(min=0,max=1)
            tt = dict(img=tt)
            tt['img_metas'] = x['img_metas']
            tt['img'] = tt['img'].to("cuda")
            tt['img'] = [tt['img']]
            out = self.model(return_loss=False, **tt)[0]
            out_l_iter = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
            out_iter = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
            miou_iter = np.nanmean(mean_iou([out_iter.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])
            print(f"miou_iter :{miou_iter}  min_miou: {min_miou} noise_range : {noise.max()}")

            if miou_iter>min_miou:
                tt = (temp['img'][0] - noise).clamp(min=0,max=1)
                tt = dict(img=tt)
                tt['img_metas'] = x['img_metas']
                tt['img'] = tt['img'].to("cuda")
                tt['img'] = [tt['img']]
                out = self.model(return_loss=False, **tt)[0]
                out_l_iter = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
                out_iter = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
                miou_iter = np.nanmean(
                    mean_iou([out_iter.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])
            rr = out_iter.clone().cpu().numpy()

            if miou_iter<=min_miou:
                min_miou = miou_iter
                min_rr=rr


        return min_miou



'''

Original License

MIT License

Copyright (c) 2020 cwllenny

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

'''
