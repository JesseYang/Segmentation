import numpy as np
from easydict import EasyDict as edict

cfg = edict()

cfg.name = 'fox'

cfg.hflip = False

# class_num should include background
cfg.class_num = 10

cfg.dilations = [1, 2, 4, 8, 16, 1, 2, 4, 8, 16]
cfg.channels = [32, 32, 32, 32, 64, 64, 64, 64, 64]
cfg.kernel_size = [3, 3, 3, 3, 3, 3, 3, 3, 3, 3]
cfg.with_bn = True

cfg.weight_decay = 5e-4

cfg.train_list = [cfg.name + "_train.txt"]
cfg.test_list = cfg.name + "_test.txt"
