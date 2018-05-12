#!/usr/bin/env python
#-*- coding:utf8 -*-

# Created by Li Bo 2018-04-23 16:28:02

import numpy as np
import _sample


### image, np.array, uint8
### rect x1, y1, x2, y2
### shape, height, width/ output
### mean mean value

def resample(img, rect, shape, mean):
    shape = map(lambda x:int(round(x)), shape)
    out = np.zeros((3, shape[0], shape[1]), np.uint8)
    rect = np.array(rect).astype(np.float32).reshape(-1)
    mean = np.array(mean).astype(np.uint8).reshape(-1)

    _sample.resample(img, rect, mean, out)
    return out



