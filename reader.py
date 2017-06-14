#!/usr/bin/env python
# -*- coding: UTF-8 -*-
# File: cifar.py
# Author: Yuxin Wu <ppwwyyxx@gmail.com>
#         Yukun Chen <cykustc@gmail.com>

import os, sys
import cv2
import pickle
import numpy as np
from scipy import misc
import struct
import six
from six.moves import urllib, range
import copy
import logging

from tensorpack import *
from cfgs.config import cfg

from morph import warp

def get_imglist(fname):
    with open(fname) as f:
        content = f.readlines()
    content = [ele.strip() for ele in content]
    return content

# def read_data(train_or_test, affine_trans=False, flip=False):
def read_data(img_path, label_path, affine_trans=False, hflip=False, scale_x=1.0, scale_y=1.0, warp_ratio=0):
    img = misc.imread(img_path, mode='RGB')
    f = open(label_path, "rb")
    binary_data = f.read()
    label = []
    # for d in binary_data:
    for idx in range(len(binary_data)):
        label.append(struct.unpack('B', binary_data[idx:idx+1])[0])
    label = np.array(label, dtype=np.uint8)

    # affine
    h, w, c = img.shape
    label = np.reshape(label, (h, w))

    if affine_trans:
        # scale_x = (np.random.uniform() - 0.5) / 4 + 1
        # scale_y = (np.random.uniform() - 0.5) / 4 + 1
        # max_offx = (scale - 1.) * w
        # max_offy = (scale - 1.) * h
        # offx = int(np.random.uniform() * max_offx)
        # offy = int(np.random.uniform() * max_offy)

        img = cv2.resize(img, (0, 0), fx=scale_x, fy=scale_y)
        # img = img[offy: (offy + h), offx: (offx + w)]

        label = cv2.resize(label, (0, 0), fx=scale_x, fy=scale_y, interpolation=cv2.INTER_NEAREST)
        # label = label[offy: (offy + h), offx: (offx + w)]

    if hflip and np.random.rand() > 0.5:
        img = cv2.flip(img, flipCode=1)
        label = cv2.flip(label, flipCode=1)

    if warp_ratio > 0 and np.random.rand() <= warp_ratio:
        img, label = warp(img, label)

    h, w, c = img.shape
    label = np.reshape(label, (h * w))

    return [img, label]

class Data(RNGDataFlow):
    def __init__(self, train_or_test, shuffle=True, affine_trans=True, hflip=True):
        assert train_or_test in ['train', 'test']
        fname_list = cfg.train_list if train_or_test == "train" else cfg.test_list
        self.train_or_test = train_or_test
        self.affine_trans = affine_trans
        self.hflip = hflip
        self.scale_x = 1.0
        self.scale_y = 1.0
        self.warp_ratio = 0.5
        fname_list = [fname_list] if type(fname_list) is not list else fname_list

        self.imglist = []
        for fname in fname_list:
            self.imglist.extend(get_imglist(fname))
        self.shuffle = shuffle

    def size(self):
        return len(self.imglist)

    def get_data(self):
        idxs = np.arange(len(self.imglist))
        if self.shuffle:
            self.rng.shuffle(idxs)
        if self.affine_trans:
            self.scale_x = (np.random.uniform() - 0.5) / 4 + 1
            self.scale_y = (np.random.uniform() - 0.5) / 4 + 1
        for k in idxs:
            img_path = self.imglist[k]
            label_path = img_path.replace('image', 'label').replace('png', 'dat')
            yield read_data(img_path, label_path, self.affine_trans, self.hflip, self.scale_x, self.scale_y, self.warp_ratio)

if __name__ == '__main__':
    ds = Data('train')
