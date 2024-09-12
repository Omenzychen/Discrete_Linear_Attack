"""
Implements the base class for decision-based black-box attacks
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mmcv
import numpy as np
import torch
from PIL import Image
from torch import Tensor as t
import openpyxl
import sys
from mmseg.core import mean_iou


class DecisionBlackBoxAttack(object):
    def __init__(self, max_queries=np.inf, epsilon=0.5, p='inf', lb=0., ub=1., batch_size=1):
        """
        :param max_queries: max number of calls to model per data point
        :param epsilon: perturbation limit according to lp-ball
        :param p: norm for the lp-ball constraint
        :param lb: minimum value data point can take in any coordinate
        :param ub: maximum value data point can take in any coordinate
        """
        assert p in ['inf', '2'], "L-{} is not supported".format(p)

        self.p = p
        self.max_queries = max_queries
        self.total_queries = 0
        self.total_successes = 0
        self.total_failures = 0
        self.total_distance = 0
        self.sigma = 0
        self.EOT = 1
        self.lb = lb
        self.ub = ub
        self.epsilon = epsilon / ub
        self.batch_size = batch_size
        self.list_loss_queries = torch.zeros(1, self.batch_size)

    def result(self):
        """
        returns a summary of the attack results (to be tabulated)
        :return:
        """
        list_loss_queries = self.list_loss_queries[1:].view(-1)
        mask = list_loss_queries > 0
        list_loss_queries = list_loss_queries[mask]
        self.total_queries = int(self.total_queries)
        self.total_successes = int(self.total_successes)
        self.total_failures = int(self.total_failures)
        return {
            "total_queries": self.total_queries,
            "total_successes": self.total_successes,
            "total_failures": self.total_failures,
            "average_num_queries": "NaN" if self.total_successes == 0 else self.total_queries / self.total_successes,
            "failure_rate": "NaN" if self.total_successes + self.total_failures == 0 else self.total_failures / (self.total_successes + self.total_failures),
            "median_num_loss_queries": "NaN" if self.total_successes == 0 else torch.median(list_loss_queries).item(),
            "config": self._config()
        }

    def _config(self):

        raise NotImplementedError

    def distance(self, x_adv, x = None):
        x_adv = torch.Tensor(x_adv)
        if x is None:
            diff = x_adv.reshape(x_adv.size(0), -1)
        else:
            diff = (x_adv - x).reshape(x.size(0), -1)
        if self.p == '2':
            out = torch.sqrt(torch.sum(diff * diff)).item()
        elif self.p == 'inf':
            out = torch.sum(torch.max(torch.abs(diff), 1)[0]).item()
        return out

    def is_adversarial(self, x, y):

        if self.targeted:
            return self.predict_label(x) == y
        else:
            return self.predict_label(x,y) != y

    def predict_label(self, xs,y):
        print(type(xs['img']))
        if type(xs['img']) is torch.Tensor:
            x_eval=dict(img=xs['img'].permute(0,1,2,3))
            x_eval['img_metas']=xs['img_metas']
        else:
            x_eval = torch.FloatTensor(xs.transpose(0,3,1,2))
        x_eval['img'] = torch.clamp(x_eval['img'], 0, 1)
        x_eval['img'] = x_eval['img'] + self.sigma * torch.randn_like(x_eval['img'])
        x_eval['img'].to("cuda")
        x_eval['img']=[x_eval['img']]

        if self.ub == 255:
            out = self.model(return_loss=False, rescale=True, **x_eval)
        else:
            out = self.model(return_loss=False, rescale=True, **x_eval)

        out_list=[]
        for i in out:
            out_list.append(torch.from_numpy(i).to('cuda'))

        label_per = torch.split(y,1,0)
        before_label_iou=[]
        for i in range(len(out_list)):
            l = out_list[i].unsqueeze(0)
            l=l.unsqueeze(0)
            before_label_iou_nanmean=np.nanmean(mean_iou([l.cpu().clone().detach().numpy()],[label_per[i].cpu().numpy()], 150, 255)[2])
            before_label_iou.append(before_label_iou_nanmean)

        return before_label_iou,x_eval

    def _perturb(self, xs_t, ys):
        raise NotImplementedError

    def run(self, xs, ys_t, model, targeted, dset, img, label,mkdir):
        self.model = model
        self.targeted = targeted
        self.train_dataset = dset
        self.logs = {
            'iteration': [0],
            'query_count': [0]
        }
        xs = xs / self.ub
        xs_t = t(xs)
        ds=dset
        ds['img']=xs_t
        before_miou, x_eval=self.predict_label(ds,ys_t)
        after_miou=self._perturb(x_eval, ys_t,img,xs,mkdir)

        return before_miou

