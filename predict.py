#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import os
import numpy as np
from scipy import misc
import argparse

from tensorpack import *

from train import Model
from cfgs.config import cfg

def predict_one(img_path, predict_func, output_path):
    img = misc.imread(img_path)
    batch_img = np.expand_dims(img, axis=0)
    predictions = predict_func([batch_img])[0]


    if cfg.class_num == 2:
        result = (1 - np.argmax(predictions, axis=3)) * 255
        mask = np.zeros(img.shape)
        mask[:,:,0] = result[0]

        output = img * 0.7 + mask * 0.3
        misc.imsave(output_path, output)
    else:
        result = np.argmax(predictions, axis=3)
        (height, width, _) = img.shape
        output = np.zeros((height,width))
        for h in range(height):
            for w in range(width):
                output[h, w] = 1.0 * result[0, h, w] / (cfg.class_num - 1)
        misc.imsave(output_path, output)
    

def predict(args):
    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["NETWORK_OUTPUT"])

    predict_func = OfflinePredictor(predict_config)

    if os.path.isfile(args.input):
        # input is a file
        predict_one(args.input, predict_func, args.output or "output.png")

    if os.path.isdir(args.input):
        # input is a directory
        output_dir = args.output or "output"
        if os.path.isdir(output_dir) == False:
            os.makedirs(output_dir)
        for (dirpath, dirnames, filenames) in os.walk(args.input):
            logger.info("Number of images to predict is " + str(len(filenames)) + ".")
            for file_idx, filename in enumerate(filenames):
                if file_idx % 10 == 0 and file_idx > 0:
                    logger.info(str(file_idx) + "/" + str(len(filenames)))
                filepath = os.path.join(args.input, filename)
                predict_one(filepath, predict_func, os.path.join(output_dir, filename))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--input', help='path to the input image', required=True)
    parser.add_argument('--output', help='path to the output image')

    args = parser.parse_args()
    predict(args)
