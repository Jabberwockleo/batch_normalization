#!/usr/bin/env python
# -*- coding: utf-8 -*-
########################################################################
#
# Copyright (c) 2017 Wan Li. All Rights Reserved
#
########################################################################

"""
File: mock_util.py
Author: Wan Li
Date: 2017/11/27 10:41:01
"""
import numpy as np
import matplotlib.pyplot as plt

def x2makeup(x_start, x_stop, bias, num):
    """Mock y=x^2

    Args:
        x_start: lower bound of domain X
        x_stop: upper bound of domain X
        bias: y = x^2 + bias
        num: number of points generated
    Returns:
        X, Y: samples
    """
    X = np.linspace(x_start, x_stop, num)
    X = np.expand_dims(X, axis=1)
    np.random.shuffle(X)
    noise = np.random.normal(loc=0.0, scale=5.0, size=X.shape)
    Y = np.square(X) + bias + noise
    return X, Y

if __name__ == "__main__":
    X, Y = x2makeup(-5, 5, 5, 1000)
    plt.scatter(X, Y)
    plt.show()
    pass
