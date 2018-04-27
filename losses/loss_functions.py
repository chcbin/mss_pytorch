# -*- coding: utf-8 -*-
__author__ = 'S.I. Mimilakis'
__copyright__ = 'MacSeNet'

# imports
import torch


def kullback_leibler(x, x_hat):
    # Generalized KL
    rec = torch.sum(x * (torch.log(x + 1e-6) - torch.log(x_hat + 1e-6)) + (x_hat - x), dim=-1)
    return torch.mean(rec)


def p_power_error(x, x_hat, p):
    if p == 2.:
        err = (x - x_hat).pow(p)
    else:
        err = (x - x_hat).abs().pow(p)
    return err


def mse(x, x_hat):
    return torch.mean(torch.sum(torch.pow(x - x_hat, 2.), dim=-1))

# EOF
