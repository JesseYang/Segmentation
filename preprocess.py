import os
from scipy import misc
import json
from shutil import copytree, rmtree
import numpy as np
from time import time
from cfgs.config import cfg
import cv2
import augmentor

def get_data(img_path, label_path):
    img = misc.imread(img_path, mode = 'L')
    mask = misc.imread(label_path, mode = 'L')
    if img is not None and mask is not None:
        IA = augmentor.ImageAugmentor()
        img, mask = IA.augment([img, mask])
        # thresholding mask
        retval, mask = cv2.threshold(mask, 50, 255, cv2.THRESH_BINARY)
        return img, mask
    return None, None

raw_data_dir = 'raw_data'
data_dir_root = 'data'

# save old data and clear the data directory
if os.path.isdir(data_dir_root):
    copytree(data_dir_root, data_dir_root + ' - ' + str(time()))
    rmtree(data_dir_root)
os.mkdir(data_dir_root)
img_dir = os.path.join(data_dir_root, 'images')
label_dir = os.path.join(data_dir_root, 'labels')
label_img_dir = os.path.join(data_dir_root, 'label_images')
os.mkdir(img_dir)
os.mkdir(label_dir)
os.mkdir(label_img_dir)

# record paths of data
records = []

# the number of generations per image
data_per_img = 10
img_names = os.listdir(raw_data_dir)
for img_name in img_names:
    # only fetch the original image
    if "mask" in img_name:
        continue

    img_path = os.path.join(raw_data_dir, img_name)
    mask_path = img_path.replace('jpg', 'mask.0.jpg')
    for i in range(data_per_img):
        img, label_img = get_data(img_path, mask_path)
        if img is None or label_img is None:
            continue

        new_img_path = os.path.join(img_dir, img_name).replace('jpg', '%d.jpg'%i)
        misc.imsave(new_img_path, img)

        label_img_path = os.path.join(label_img_dir, img_name).replace('jpg', '%d.png'%i)
        misc.imsave(label_img_path, label_img)

        label_path = os.path.join(label_dir, img_name).replace('jpg', '%d.dat'%i)
        label_file = open(label_path, "wb")
        label_img.resize((1,label_img.size))

        #label_data = np.ones(img.shape[0:2], dtype='byte')

        label_img = list(label_img[0])
        label_img = [int(1) if i==0 else 2 for i in label_img]
        byte_data = bytearray(label_img)
        label_file.write(byte_data)
        label_file.close()
        # record the path
        records.append(new_img_path + "\n")

# split into training set and test set
test_ratio = 0.1
total_num = len(records)
test_num = int(test_ratio * total_num)
train_num = total_num - test_num
train_records = records[0:train_num]
test_records = records[train_num:]

# save to text file
all_out_file = open('all.txt', "w")
for record in records:
    all_out_file.write(record)
all_out_file.close()

train_out_file = open(cfg.train_list, "w")
for record in train_records:
    train_out_file.write(record)
train_out_file.close()

test_out_file = open(cfg.test_list, "w")
for record in test_records:
    test_out_file.write(record)
test_out_file.close()
