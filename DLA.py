from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import mmcv
import numpy as np
import torch
import torchvision.transforms
from PIL import Image
from torch import Tensor as t, Tensor
import torch.nn.functional as F
from scipy.ndimage import filters
from attacks.decision.decision_black_box_attack import DecisionBlackBoxAttack
from mmseg.core import mean_iou
from utils.compute import lp_step, sign
import cv2


class DLA(DecisionBlackBoxAttack):
    """
    NES Attack
    """

    def __init__(self, epsilon, batch_size,name,T):
        """
        :param max_loss_queries: maximum number of calls allowed to loss oracle per data pt
        :param epsilon: radius of lp-ball of perturbation
        :param p: specifies lp-norm  of perturbation
        :param fd_eta: forward difference step
        :param lr: learning rate of NES step
        :param q: number of noise samples per NES step
        :param lb: data lower bound
        :param ub: data upper bound
        """
        super().__init__(epsilon=epsilon,
                         batch_size=batch_size)
        self.T=T
        self.name=name
        self.i=0
        self.h=0
        self.best_est_deriv=0
        self.sign_mask = None



    def _perturb(self, x, y, img, xs,mkdir):
        c, h, w = x['img'][0].shape[1:]

        _shape = list(x['img'][0].shape)
        dim = np.prod(_shape[1:])
        num_axes = len(_shape[1:])
        gs_t = torch.zeros_like(x['img'][0])
        out = self.model(return_loss=False, **x)[0]
        out_l = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
        before_miou = np.nanmean(mean_iou([out_l.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])

        out_l_iter = torch.from_numpy(out).to('cuda').unsqueeze(0).unsqueeze(0)
        min_rr = out_l_iter.clone().cpu().numpy()

        attack_x = x['img'][0].clone()
        attack_x = dict(img=attack_x)
        attack_x['img_metas'] = x['img_metas']
        attack_x['img'] = attack_x['img'].to("cuda")
        attack_x['img'] = [attack_x['img']]


        temp = x['img'][0].clone()
        temp = dict(img=temp)
        temp['img_metas'] = x['img_metas']
        temp['img'] = temp['img'].to("cuda")
        temp['img'] = [temp['img']]

        miou_iter = before_miou
        min_miou = before_miou
        f_miou=min_miou

        ori_patch_noise = torch.zeros(1, 3, h, w)

        istart=0
        iend=0
        if self.i == 0 and self.h == 0:
            self.sign_mask = sign(torch.ones(1, 3, h, w))

        for iter in range(self.T):
            gs_t = torch.zeros_like(x['img'][0])
            patch_noise = torch.zeros(1, 3, h, w)
            c, h, w = x['img'][0].shape[1:]
            square_x = temp['img'][0].clone()
            flag_direction=0

            if iter<(self.T)/5:
                if iter%2==0:
                    init_delta = t(np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], 1, h, 1])).repeat(1,3,1,w)
                    xi=(temp['img'][0]+init_delta).clamp(min=0,max=1)
                    xi = dict(img=xi)
                    xi['img_metas'] = x['img_metas']
                    xi['img'] = xi['img'].to('cuda')
                    xi['img'] = [xi['img']]

                    fxs_t_res = self.model(return_loss=False, **xi)[0]
                    f_l = torch.from_numpy(fxs_t_res).to('cuda').unsqueeze(0).unsqueeze(0)
                    f_miou = np.nanmean(mean_iou([f_l.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])

                    if f_miou<=min_miou:
                        flag_direction=0
                        ori_patch_noise=init_delta


                else:
                    init_delta = t(np.random.choice([-self.epsilon, self.epsilon], size=[xs.shape[0], 1, 1, w])).repeat(1,3,h,1)
                    xi = (temp['img'][0] + init_delta).clamp(min=0, max=1)
                    xi = dict(img=xi)
                    xi['img_metas'] = x['img_metas']
                    xi['img'] = xi['img'].to('cuda')
                    xi['img'] = [xi['img']]

                    fxs_t_res = self.model(return_loss=False, **xi)[0]
                    f_l = torch.from_numpy(fxs_t_res).to('cuda').unsqueeze(0).unsqueeze(0)
                    f_miou = np.nanmean(mean_iou([f_l.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])

                    if f_miou <= min_miou:
                        flag_direction = 1
                        ori_patch_noise = init_delta


            else:

                if iter==(self.T)/5:

                    delta = self.sign_mask * ori_patch_noise
                    xi = attack_x['img'][0]
                    memp = xi
                    xi=(xi+delta).clamp(min=0,max=1)
                    xi = dict(img=xi)
                    xi['img_metas'] = x['img_metas']
                    xi['img'] = xi['img'].to('cuda')
                    xi['img'] = [xi['img']]
                    fxs_t_res = self.model(return_loss=False, **xi)[0]
                    f_l = torch.from_numpy(fxs_t_res).to('cuda').unsqueeze(0).unsqueeze(0)
                    f_miou = np.nanmean(mean_iou([f_l.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])
                    est_deriv=f_miou-min_miou
                    self.best_est_deriv=(est_deriv>self.best_est_deriv) * self.best_est_deriv+(est_deriv<=self.best_est_deriv)*est_deriv


                else:
                    if flag_direction==1:
                        dim=w
                        c_len=np.ceil(w/(2**self.h)).astype(int)
                        istart=self.i*c_len
                        iend=min(w,(self.i+1)*c_len)
                        self.sign_mask[:,:,:,istart:iend]*=-1.
                        delta = self.sign_mask * ori_patch_noise
                        xi = attack_x['img'][0]

                        memp = xi
                        xi = (xi + delta).clamp(min=0,max=1)
                        xi = dict(img=xi)
                        xi['img_metas'] = x['img_metas']
                        xi['img'] = xi['img'].to('cuda')
                        xi['img'] = [xi['img']]
                        fxs_t_res = self.model(return_loss=False, **xi)[0]
                        f_l = torch.from_numpy(fxs_t_res).to('cuda').unsqueeze(0).unsqueeze(0)
                        f_miou = np.nanmean(
                            mean_iou([f_l.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])
                        est_deriv = f_miou - min_miou
                        self.best_est_deriv = (est_deriv > self.best_est_deriv) * self.best_est_deriv + (
                                    est_deriv <= self.best_est_deriv) * est_deriv
                        if est_deriv >= self.best_est_deriv:
                            self.sign_mask[:,:,:,istart:iend]*=-1.



                    else:

                        dim=h
                        c_len = np.ceil(h / (2 ** self.h)).astype(int)
                        istart = self.i * c_len
                        iend = min(h, (self.i + 1) * c_len)
                        self.sign_mask[:, :, istart:iend,:] *= -1.
                        delta = self.sign_mask * ori_patch_noise
                        xi = attack_x['img'][0]
                        memp = xi
                        xi = (xi + delta).clamp(min=0,max=1)
                        xi = dict(img=xi)
                        xi['img_metas'] = x['img_metas']
                        xi['img'] = xi['img'].to('cuda')
                        xi['img'] = [xi['img']]
                        fxs_t_res = self.model(return_loss=False, **xi)[0]
                        f_l = torch.from_numpy(fxs_t_res).to('cuda').unsqueeze(0).unsqueeze(0)
                        f_miou = np.nanmean(
                            mean_iou([f_l.cpu().clone().detach().numpy()], [y.cpu().numpy()], 150, 255)[2])
                        est_deriv = f_miou - min_miou
                        self.best_est_deriv = (est_deriv > self.best_est_deriv) * self.best_est_deriv + (
                                est_deriv <= self.best_est_deriv) * est_deriv
                        if est_deriv >= self.best_est_deriv:
                            self.sign_mask[:, :, istart:iend,:] *= -1.


                    self.i += 1
                    if self.i == 2 ** self.h or (iend == h and flag_direction == 1) or (iend == w and flag_direction == 0):
                        self.h += 1
                        self.i = 0
                        if self.h == np.ceil(np.log2(dim)).astype(int) + 1:
                            ori_patch_noise = ori_patch_noise * self.sign_mask
                            self.h = 0
            out_l_iter = torch.from_numpy(fxs_t_res).to('cuda').unsqueeze(0).unsqueeze(0)
            rr = out_l_iter.clone().cpu().numpy()
            if f_miou < min_miou:
                min_miou = f_miou
                min_rr = rr

        self.sign_mask=None
        self.i=0
        self.h=0
        self.best_est_deriv = 0
        return min_miou

    def _config(self):
        return {
            "name": self.name,
            "epsilon": self.epsilon,
            "attack_name": self.__class__.__name__
        }


'''

Original License

MIT License

Copyright (c) 2019 Abdullah Al-Dujaili

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
