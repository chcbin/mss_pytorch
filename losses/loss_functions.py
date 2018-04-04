# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch


def kullback_leibler(x, x_hat):
    # Generalized KL
    rec = torch.sum(x * (torch.log(x + 1e-6) - torch.log(x_hat + 1e-6)) + (x_hat - x), dim=-1)
    return torch.mean(rec)


def mse(x, x_hat):
    return torch.mean(torch.pow(x - x_hat, 2.))


def mt_l1_based(x, x_hat, masking_treshold):
    err = torch.abs(x - x_hat) * masking_treshold
    return torch.mean(err)


def mt_l2_based(x, x_hat, masking_treshold):
    err = torch.pow(x - x_hat, 2.) * masking_treshold
    return torch.mean(err)

# EOF
