import scipy.ndimage as ndi
from scipy import misc
import cv2
from random import randrange, randint
import math
import numpy as np

class ImageAugmentor():
    def __init__(self):
        """
        # Arguments
            max_deg
        """
        self.max_deg = 20
        self.min_vignetting = 400
        self.max_vignetting = 1000
        self.crop_w = 400
        self.crop_h = 400


    def largest_rotated_rect(self, w, h, angle):
        """
        Get largest rectangle after rotation.
        http://stackoverflow.com/questions/16702966/rotate-image-and-crop-out-black-borders
        """
        angle = angle / 180.0 * math.pi
        if w <= 0 or h <= 0:
            return 0, 0

        width_is_longer = w >= h
        side_long, side_short = (w, h) if width_is_longer else (h, w)

        # since the solutions for angle, -angle and 180-angle are all the same,
        # if suffices to look at the first quadrant and the absolute values of sin,cos:
        sin_a, cos_a = abs(math.sin(angle)), abs(math.cos(angle))
        if side_short <= 2. * sin_a * cos_a * side_long:
            # half constrained case: two crop corners touch the longer side,
            #   the other two corners are on the mid-line parallel to the longer line
            x = 0.5 * side_short
            wr, hr = (x / sin_a, x / cos_a) if width_is_longer else (x / cos_a, x / sin_a)
        else:
            # fully constrained case: crop touches all 4 sides
            cos_2a = cos_a * cos_a - sin_a * sin_a
            wr, hr = (w * cos_a - h * sin_a) / cos_2a, (h * cos_a - w * sin_a) / cos_2a

        return int(np.round(wr)), int(np.round(hr))

    def random_rotate_and_crop_valid(self, imgs):
        """Randomly rotate image and crop the largest rectangle inside the rotated image.

        # Arguments
            imgs: list of processing images.
        """
        if type(imgs) is not list:
            imgs = [imgs]
        
        deg = randrange(-self.max_deg, self.max_deg)
        
        res = []
        for img in imgs:
            center = (img.shape[1] * 0.5, img.shape[0] * 0.5)
            rot_m = cv2.getRotationMatrix2D((center[0] - 0.5, center[1] - 0.5), deg, 1)
            ret = cv2.warpAffine(img, rot_m, img.shape[1::-1])
            if img.ndim == 3 and ret.ndim == 2:
                ret = ret[:, :, np.newaxis]
            neww, newh = self.largest_rotated_rect(ret.shape[1], ret.shape[0], deg)
            neww = min(neww, ret.shape[1])
            newh = min(newh, ret.shape[0])
            newx = int(center[0] - neww * 0.5)
            newy = int(center[1] - newh * 0.5)
            # print(ret.shape, deg, newx, newy, neww, newh)
            res.append(ret[newy:newy + newh, newx:newx + neww])
        
        return res

    def random_crop(self, imgs):
        """Randomly crop the image into a smaller one.

        # Arguments
            imgs: list of processing images.

        """
        if type(imgs) is not list:
            imgs = [imgs]
        
        h, w = imgs[0].shape
        y = randint(0,h - self.crop_h - 1)
        x = randint(0,w - self.crop_w - 1)
        res = []
        for img in imgs:
            ret = img[y:y + self.crop_h, x:x + self.crop_w]
            res.append(ret)
        
        return res

    def vignetting(self,img):
        """Perfo
        """

        h, w = img.shape
        h_gk_size = randint(self.min_vignetting,self.max_vignetting)
        w_gk_size = randint(self.min_vignetting,self.max_vignetting)
        h_gk = cv2.getGaussianKernel(h,h_gk_size)
        w_gk = cv2.getGaussianKernel(w,w_gk_size)
        c = h_gk*w_gk.T
        d = c/c.max()
        e = img*d
        
        return e

    def gaussian_blur(self, img):
        kernel_size = (19, 19)
        sigma = 1.5
        img = cv2.GaussianBlur(img, kernel_size, sigma)
        return img
    
    def augment(self, imgs):
        """Augment all input images with a given procedure
        """
        img, label = imgs

        # vignetting
        img = self.vignetting(img)
        # gaussian_blur
        img = self.gaussian_blur(img)
        # rotate & crop
        img, label = self.random_rotate_and_crop_valid([img,label])
        # randomly crop into same size
        img, label = self.random_crop([img,label])

        return img, label

if __name__ == '__main__':
    IA = ImageAugmentor()
    img = misc.imread('1.jpg', mode = 'L')
    label = misc.imread('1.mask.0.jpg', mode = 'L')
    img,label = IA.augment([img,label])
    misc.imsave('output.jpg', img)
    misc.imsave('label.jpg', label)