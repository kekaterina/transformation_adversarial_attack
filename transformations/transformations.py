import sys

sys.path.append('../')

from bagnet import *
import cv2 as cv
import numpy as np


TRANSFORMATION_PARAMS = {'scale': {'min': 0.9, 'max': 1.4},
                         'rotation': {'min': -22.5, 'max': 22.5},
                         'dark': {'min': -0.05, 'max': 0.05}
                         }


def reshape_image(image):
    new_image = np.moveaxis(image, 0, 2)
    return new_image


def rotate_image(image, angle, scale, center):
    if center is None:
        center = (image.shape[1] // 2, image.shape[0] // 2)
    rot_mat = cv.getRotationMatrix2D(center, angle, scale)
    warp_rotate_dst = cv.warpAffine(image, rot_mat, (image.shape[1], image.shape[0]))

    return warp_rotate_dst


def dark_image(image, delta):
    dark_src = image - delta
    return dark_src


def gauss_transformer(img):
    gaussian_trans = np.random.normal(0.0, 0.1, img.shape)
    return img + gaussian_trans


def get_transformation_choise(shape, transformation_params=TRANSFORMATION_PARAMS):
    scale_trans = np.random.uniform(transformation_params['scale']['min'],
                                    transformation_params['scale']['max'],
                                    10000)
    rotation_trans = np.random.uniform(transformation_params['rotation']['min'],
                                       transformation_params['rotation']['max'],
                                       10000)
    dark_trans = np.random.uniform(transformation_params['dark']['min'],
                                   transformation_params['dark']['max'],
                                   10000)

    return {'scale_trans': scale_trans,
            'rotation_trans': rotation_trans,
            'dark_trans': dark_trans}