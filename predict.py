#!/usr/bin/env python
# -*- coding: utf-8 -*-
# File: train.py
# Author: Jesse Yang <jesse.yang1985@gmail.com>

import os
import numpy as np
import pdb
from scipy import misc
import argparse
import pydensecrf.densecrf as dcrf
from pydensecrf.utils import unary_from_labels, unary_from_softmax, create_pairwise_bilateral, create_pairwise_gaussian
import ntpath

from tensorpack import *

from train import Model
from cfgs.config import cfg

def predict_one(img_path, predict_func, output_path, crf):
    img = misc.imread(img_path)
    batch_img = np.expand_dims(img, axis=0)
    predictions = predict_func([batch_img])[0]

    predictions = np.reshape(predictions, (img.shape[0], img.shape[1], cfg.class_num))

    # conditional random field
    if crf is True:
        d = dcrf.DenseCRF2D(img.shape[1], img.shape[0], cfg.class_num)

        # set unary potential
        predictions = np.transpose(predictions, (2, 0, 1))
        U = unary_from_softmax(predictions)
        d.setUnaryEnergy(U)

        # set pairwise potential
        # This creates the color-independent features and then add them to the CRF
        d.addPairwiseGaussian(sxy=(3, 3), compat=3, kernel=dcrf.DIAG_KERNEL,
                              normalization=dcrf.NORMALIZE_SYMMETRIC)
        # This adds the color-dependent term, i.e. features are (x,y,r,g,b).
        d.addPairwiseBilateral(sxy=(8, 8), srgb=(13, 13, 13), rgbim=img,
                               compat=10,
                               kernel=dcrf.DIAG_KERNEL,
                               normalization=dcrf.NORMALIZE_SYMMETRIC)

        iter_num = 5
        result = np.argmax(d.inference(iter_num), axis=0)
        result = np.reshape(result, (img.shape[0], img.shape[1]))
    else:
        result = np.argmax(predictions, axis=2)

    if cfg.class_num == 2:
        result = (1 - result) * 255
        mask = np.zeros(img.shape)
        mask[:,:,0] = result
        output = img * 0.7 + mask * 0.3
        misc.imsave(output_path, output)
    else:
        (height, width, _) = img.shape
        output = np.zeros((height,width))
        for h in range(height):
            for w in range(width):
                output[h, w] = 1.0 * result[h, w] / (cfg.class_num - 1)
        misc.imsave(output_path, output)
    

def predict(args):
    sess_init = SaverRestore(args.model)
    model = Model()
    predict_config = PredictConfig(session_init=sess_init,
                                   model=model,
                                   input_names=["input"],
                                   output_names=["softmax_output"])

    predict_func = OfflinePredictor(predict_config)

    if os.path.isfile(args.input):
        if args.input.endswith("txt"):
            # input is an text file
            with open(args.input) as f:
                records = f.readlines()
            records = [e.strip() for e in records]
            output_dir = args.output or "output"
            for record in records:
                filename = ntpath.basename(record)
                predict_one(record, predict_func, os.path.join(output_dir, filename), args.crf)
        else:
            # input is an image file
            predict_one(args.input, predict_func, args.output or "output.png", args.crf)

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
                predict_one(filepath, predict_func, os.path.join(output_dir, filename), args.crf)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help='path to the model file', required=True)
    parser.add_argument('--input', help='path to the input image', required=True)
    parser.add_argument('--output', help='path to the output image/dir')
    parser.add_argument('--crf', action='store_true')


    args = parser.parse_args()
    predict(args)
